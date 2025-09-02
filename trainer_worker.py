import grpc
import time
import io

import numpy as np

from concurrent import futures
from threading import Thread
from PIL import Image
from typing import List, Tuple, Iterable
from weightslab.experiment import ArchitectureOpType
import torch
import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from collections import defaultdict
from torchvision import transforms
from scope_timer import ScopeTimer

# from fashion_mnist_exp import get_exp

# from hct_kaggle_exp import get_exp
# from cifar_exp import get_exp
from imagenet_exp import get_exp
# from imagenet_exp_deep import get_exp
# from imagenet_convnext import get_exp
# from mnist_exp_fully_conv import get_exp
# from imagenet_effnet_exp import get_exp
# from segmentation_exp import get_exp
# from cad_models_exp import get_exp

experiment = get_exp()
# experiment.set_is_training(True)


def training_thread_callback():
    while True:
        # print("Training thread callback ", str(experiment), end="\r")
        if experiment.get_is_training():
            experiment.train_step_or_eval_full()
            # print(f"[TRAIN] Steps left: {experiment.get_training_steps_to_do()}")


training_thread = Thread(target=training_thread_callback)
training_thread.start()


HYPER_PARAMETERS = {
    ("Experiment Name", "experiment_name", "text", lambda: experiment.name),
    ("Left Training Steps", "training_left", "number", lambda: experiment.training_steps_to_do),
    ("Learning Rate", "learning_rate", "number", lambda: experiment.learning_rate),
    ("Batch Size", "batch_size", "number", lambda: experiment.batch_size),
    ("Eval Frequency", "eval_frequency", "number", lambda: experiment.eval_full_to_train_steps_ratio),
    ("Checkpoint Frequency", "checkpooint_frequency", "number", lambda: experiment.experiment_dump_to_train_steps_ratio),
}


def get_hyper_parameters_pb(
        hype_parameters_desc_tuple: Tuple) -> List[pb2.HyperParameterDesc]:

    hyper_parameters_pb2 = []
    for (label, name, type_, getter) in hype_parameters_desc_tuple:
        hyper_parameter_pb2 = None
        if type_ == "text":
            hyper_parameter_pb2 = pb2.HyperParameterDesc(
                label=label,
                name=name,
                type=type_,
                string_value=getter()
            )
        else:
            hyper_parameter_pb2 = pb2.HyperParameterDesc(
                label=label,
                name=name,
                type=type_,
                numerical_value=getter()
            )
        hyper_parameters_pb2.append(hyper_parameter_pb2)

    return hyper_parameters_pb2


def get_neuron_representations(layer) -> Iterable[pb2.NeuronStatistics]:

    layer_id = layer.get_module_id()
    neuron_representations = []
    for neuron_idx in range(layer.neuron_count):
        age = int(layer.train_dataset_tracker.get_neuron_age(neuron_idx))
        trate = layer.train_dataset_tracker.get_neuron_triggers(neuron_idx)
        erate = layer.eval_dataset_tracker.get_neuron_triggers(neuron_idx)
        evage = layer.eval_dataset_tracker.get_neuron_age(neuron_idx)

        trate = trate/age if age > 0 else 0
        erate = erate/evage if evage > 0 else 0

        neuron_lr = layer.get_per_neuron_learning_rate(neuron_idx)

        neuron_representation = pb2.NeuronStatistics(
            neuron_id=pb2.NeuronId(layer_id=layer_id, neuron_id=neuron_idx),
            neuron_age=age,
            train_trigger_rate=trate,
            eval_trigger_rate=erate,
            learning_rate=neuron_lr,
        )
        for incoming_id, incoming_lr in  layer.incoming_neuron_2_lr.items():
            neuron_representation.incoming_neurons_lr[incoming_id] = incoming_lr

        neuron_representations.append(neuron_representation)

    return neuron_representations


def get_layer_representation(layer) -> pb2.LayerRepresentation:
    layer_representation = None
    if "Conv2d" in layer.__class__.__name__:
        layer_representation = pb2.LayerRepresentation(
            layer_id=layer.get_module_id(),
            layer_name=layer.__class__.__name__,
            layer_type="Conv2d",
            incoming_neurons_count=layer.incoming_neuron_count,
            neurons_count=layer.neuron_count,
            kernel_size=layer.kernel_size[0],
            stride=layer.stride[0],
        )
    elif "Linear" in layer.__class__.__name__:
        layer_representation = pb2.LayerRepresentation(
            layer_id=layer.get_module_id(),
            layer_name=layer.__class__.__name__,
            layer_type="Linear",
            incoming_neurons_count=layer.incoming_neuron_count,
            neurons_count=layer.neuron_count,
        )
    if layer_representation is None:
        return None

    layer_representation.neurons_statistics.extend(
        get_neuron_representations(layer))
    return layer_representation


def get_layer_representations(model):
    layer_representations = []
    for layer in model.layers:
        layer_representation = get_layer_representation(layer)
        if layer_representation is None:
            continue
        layer_representations.append(layer_representation)
    return layer_representations

def make_task_field(name, value):
    if isinstance(value, float):
        return pb2.TaskField(name=name, float_value=value)
    elif isinstance(value, int):
        return pb2.TaskField(name=name, int_value=value)
    elif isinstance(value, str):
        return pb2.TaskField(name=name, string_value=value)
    elif isinstance(value, bytes):
        return pb2.TaskField(name=name, bytes_value=value)
    elif isinstance(value, bool):
        return pb2.TaskField(name=name, bool_value=value)
    else:
        raise ValueError(f"Unsupported value type for TaskField: {name}: {type(value)}")


def mask_to_png_bytes(mask, num_classes=21):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    mask = np.squeeze(mask)
    if mask.ndim == 1:
        sz = int(np.sqrt(mask.size))
        if sz * sz == mask.size:
            mask = mask.reshape((sz, sz))
        else:
            raise ValueError(f"Cannot reshape mask of size {mask.size} to square.")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask 2D, got shape {mask.shape}")

    mask = (mask.astype(np.float32) * (255.0 / (num_classes - 1))).astype(np.uint8)

    im = Image.fromarray(mask)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def get_data_set_representation(dataset) -> pb2.SampleStatistics:
    # print("[BACKEND].get_data_set_representation")

    all_rows = list(dataset.as_records())

    sample_stats = pb2.SampleStatistics()
    sample_stats.sample_count = len(dataset.wrapped_dataset)
    task_type = getattr(experiment, "task_type", "classification")
    sample_stats.task_type = task_type

    for sample_id, row in enumerate(all_rows):
        target = row.get("label", row.get("target", -1))
        pred = row.get("prediction_raw", -1)

        if task_type == "classification":
            target_list = [int(target)] if not isinstance(target, (list, np.ndarray)) else [int(np.array(target).item())]
            pred_list = [int(pred)] if not isinstance(pred, (list, np.ndarray)) else [int(np.array(pred).item())]
        elif task_type == "segmentation":
            if hasattr(target, 'cpu'):  
                target_arr = target.cpu().numpy()
            else:
                target_arr = np.array(target)
            if hasattr(pred, 'cpu'):
                pred_arr = pred.cpu().numpy()
            else:
                pred_arr = np.array(pred)
            target_classes = np.unique(target_arr)
            pred_classes = np.unique(pred_arr)
            target_list = [int(x) for x in target_classes]
            pred_list = [int(x) for x in pred_classes]

        else:
            target_list = [int(target)] if not isinstance(target, (list, np.ndarray)) else [int(np.array(target).item())]
            pred_list = [int(pred)] if not isinstance(pred, (list, np.ndarray)) else [int(np.array(pred).item())]
        record = pb2.RecordMetadata(
            sample_id=row.get('sample_id', sample_id),
            sample_label=target_list,
            sample_prediction=pred_list,
            sample_last_loss=float(row.get('prediction_loss', -1)),
            sample_encounters=int(row.get('encountered', row.get('exposure_amount', 0))),
            sample_discarded=bool(row.get('deny_listed', False)),
        )
        sample_stats.records.append(record)

    return sample_stats


def tensor_to_bytes(tensor):
    # Convert tensor to numpy array and transpose to (H, W, C) format
    if tensor.shape[0] > 1:
        np_img = tensor.numpy().transpose(1, 2, 0)
        np_img = (np_img * 255).astype(np.uint8)
        mode = "RGB"
    else:
        np_img = tensor.squeeze(0).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        mode = "L"

    img = Image.fromarray(np_img, mode=mode)
    buf = io.BytesIO()
    # img.save(buf, format='png')
    img.save(buf, format='jpeg', quality=85)
    return buf.getvalue()

def load_raw_image(dataset, index):

    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    if hasattr(wrapped, "images") and isinstance(wrapped.images, list):
        img_path = wrapped.images[index]
        img = Image.open(img_path)
        return img.convert("RGB")
    elif hasattr(wrapped, "data"):
        np_img = wrapped.data[index]
        if hasattr(np_img, 'numpy'):
            np_img = np_img.numpy()  
        if np_img.ndim == 2:
            return Image.fromarray(np_img.astype(np.uint8), mode="L")
        elif np_img.ndim == 3:
            return Image.fromarray(np_img.astype(np.uint8), mode="RGB")
        else:
            raise ValueError(f"Unsupported image shape: {np_img.shape}")

    elif hasattr(wrapped, "samples") or hasattr(wrapped, "imgs"):
        if hasattr(wrapped, "samples"):
            img_path, _ = wrapped.samples[index]
        else:
            img_path, _ = wrapped.imgs[index]
        img = Image.open(img_path)
        return img.convert("L") if img.mode in ["1", "L", "I;16", "I"] else img.convert("RGB")

    else:
        raise ValueError("Dataset type not supported for raw image extraction.")


def _get_input_tensor_for_sample(dataset, sample_id):
    if hasattr(dataset, "_getitem_raw"):
        tensor, _, _ = dataset._getitem_raw(sample_id)
    else:
        tensor, _ = dataset[sample_id]

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0) 
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ExperimentServiceServicer(pb2_grpc.ExperimentServiceServicer):
    def StreamStatus(self, request_iterator, context):
        global experiment
        while True:
            log = experiment.logger.queue.get()
            if "metric_name" in log and "acc" in log["metric_name"]:
                # print(f"[LOG] {log['metric_name']} = {log['metric_value']:.2f}")
                pass

            if log is None:
                break
            metrics_status, annotat_status = None, None
            if "metric_name" in log:
                metrics_status = pb2.MetricsStatus(
                    name=log["metric_name"],
                    value=log["metric_value"],
                )
            elif "annotation" in log:
                annotat_status = pb2.AnnotatStatus(
                    name=log["annotation"])
                for key, value in log["metadata"].items():
                    annotat_status.metadata[key] = value

            training_status = pb2.TrainingStatusEx(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                experiment_name=log["experiment_name"],
                model_age=log["model_age"],
            )

            if metrics_status:
                training_status.metrics_status.CopyFrom(metrics_status)
            if annotat_status:
                training_status.annotat_status.CopyFrom(annotat_status)

            experiment.logger.queue.task_done()

            yield training_status

    def ExperimentCommand(self, request, context):
        # print("ExperimentServiceServicer.ExperimentCommand", request)
        if request.HasField('hyper_parameter_change'):
            # TODO(rotaru): handle this request
            hyper_parameters = request.hyper_parameter_change.hyper_parameters

            if hyper_parameters.HasField('is_training'):
                experiment.set_is_training(hyper_parameters.is_training)

            if hyper_parameters.HasField('learning_rate'):
                experiment.set_learning_rate(
                    hyper_parameters.learning_rate)

            if hyper_parameters.HasField('batch_size'):
                experiment.set_batch_size(hyper_parameters.batch_size)

            if hyper_parameters.HasField('training_steps_to_do'):
                experiment.set_training_steps_to_do(
                    hyper_parameters.training_steps_to_do)

            if hyper_parameters.HasField('experiment_name'):
                experiment.set_name(hyper_parameters.experiment_name)
            return pb2.CommandResponse(
                success=True, message="Hyper parameter changed")
        if request.HasField('deny_samples_operation'):
            denied_cnt = len(request.deny_samples_operation.sample_ids)
            experiment.train_loader.dataset.denylist_samples(
                set(request.deny_samples_operation.sample_ids),
                accumulate = request.deny_samples_operation.accumulate
            )
            return pb2.CommandResponse(
                success=True, message=f"Denied {denied_cnt} train samples")
        if request.HasField('deny_eval_samples_operation'):
            denied_cnt = len(request.deny_eval_samples_operation.sample_ids)
            experiment.eval_loader.dataset.denylist_samples(
                set(request.deny_eval_samples_operation.sample_ids),
                accumulate = request.deny_eval_samples_operation.accumulate
            )
            return pb2.CommandResponse(
                success=True, message=f"Denied {denied_cnt} eval samples")

        if request.HasField('remove_from_denylist_operation'):
            allowed = set(request.remove_from_denylist_operation.sample_ids)
            experiment.train_loader.dataset.allowlist_samples(allowed)
            return pb2.CommandResponse(
                success=True, message=f"Un-denied {len(allowed)} train samples")

        if request.HasField('remove_eval_from_denylist_operation'):
            allowed = set(request.remove_eval_from_denylist_operation.sample_ids)
            experiment.eval_loader.dataset.allowlist_samples(allowed)
            return pb2.CommandResponse(
                success=True, message=f"Un-denied {len(allowed)} eval samples")

        if request.HasField('load_checkpoint_operation'):
            checkpoint_id = request.load_checkpoint_operation.checkpoint_id
            experiment.load(checkpoint_id)

        response = pb2.CommandResponse(success=True, message="")
        if request.get_hyper_parameters:
            response.hyper_parameters_descs.extend(
                get_hyper_parameters_pb(HYPER_PARAMETERS))
        if request.get_interactive_layers:
            if request.HasField('get_single_layer_info_id'):
                response.layer_representations.extend([
                    get_layer_representation(
                        experiment.model.get_layer_by_id(
                            request.get_single_layer_info_id))])
            else:
                response.layer_representations.extend(
                    get_layer_representations(experiment.model))
                # print(response)
        if request.get_data_records:
            if request.get_data_records == "train":
                response.sample_statistics.CopyFrom(
                    get_data_set_representation(
                        experiment.train_loader.dataset))
                response.sample_statistics.origin = "train"
            elif request.get_data_records == "eval":
                response.sample_statistics.CopyFrom(
                    get_data_set_representation(
                        experiment.eval_loader.dataset))
                response.sample_statistics.origin = "eval"

        return response

    def GetSample(self, request, context):
        # print(f"ExperimentServiceServicer.GetSample({request})")

        if not request.HasField('sample_id') or not request.HasField('origin'):
            return pb2.SampleRequestResponse(
                error_message="Invalid request. Provide sample_id & origin.")

        if request.origin not in ["train", "eval"]:
            return pb2.SampleRequestResponse(
                error_message=f"Invalid origin {request.origin}")

        if request.sample_id < 0:
            return pb2.SampleRequestResponse(
                error_message=f"Invalid sample_id {request.sample_id}")

        dataset = None
        if request.origin == "train":
            dataset = experiment.train_loader.dataset
        elif request.origin == "eval":
            dataset = experiment.eval_loader.dataset

        if dataset is None:
            return pb2.SampleRequestResponse(
                error_message=f"Dataset {request.origin} not found.")

        if request.sample_id >= len(dataset):
            return pb2.SampleRequestResponse(
                error_message=f"Sample {request.sample_id} not found.")

        transformed_tensor, idx, label = dataset._getitem_raw(request.sample_id)
        # #TODO: apply transform too
        
        transformed_image_bytes = tensor_to_bytes(transformed_tensor)

        try:
            pil_img = load_raw_image(dataset, request.sample_id)
            buf = io.BytesIO()
            # pil_img.save(buf, format='PNG')
            pil_img.save(buf, format='jpeg', quality=85)
            raw_image_bytes = buf.getvalue()
        except Exception as e:
            return pb2.SampleRequestResponse(error_message=str(e))

        response = pb2.SampleRequestResponse(
            sample_id=request.sample_id,
            origin=request.origin,
            label=label,
            raw_data=raw_image_bytes,         
            data=transformed_image_bytes, 
        )

        return response
    
    def GetSamples(self, request, context):
        dataset = experiment.train_loader.dataset if request.origin == "train" else experiment.eval_loader.dataset
        response = pb2.BatchSampleResponse()

        do_resize = request.HasField("resize_width") and request.HasField("resize_height")
        resize_dims = (request.resize_width, request.resize_height) if do_resize else None
        task_type = getattr(dataset, "task_type", getattr(experiment, "task_type", "classification"))

        for sid in request.sample_ids:
            try:
                if hasattr(dataset, "_getitem_raw"):
                    tensor, idx, label = dataset._getitem_raw(sid)
                else:
                    tensor, idx, label = dataset[sid]

                if isinstance(tensor, torch.Tensor):
                    img = tensor.detach().cpu()
                else:
                    img = torch.tensor(tensor)
                if img.ndim == 3:
                    pil_img = transforms.ToPILImage()(img)
                elif img.ndim == 2:
                    pil_img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
                else:
                    raise ValueError("Unknown image shape.")

                if resize_dims:
                    pil_img = pil_img.resize(resize_dims, Image.BILINEAR)
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                transformed_bytes = buf.getvalue()

                try:
                    raw = load_raw_image(dataset, sid)
                    if resize_dims:
                        raw = raw.resize(resize_dims, Image.BILINEAR)
                    raw_buf = io.BytesIO()
                    raw.save(raw_buf, format='PNG')
                    raw_bytes = raw_buf.getvalue()
                except Exception:
                    raw_bytes = transformed_bytes 

                mask_bytes = b""
                pred_bytes = b""

                if task_type == "segmentation":
                    mask_bytes = mask_to_png_bytes(label)
                    pred_mask = dataset.get_prediction_mask(sid)
                    pred_bytes = mask_to_png_bytes(pred_mask)


                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    label=int(label) if task_type == "classification" else -1,  # not used for segmentation
                    data=transformed_bytes,
                    raw_data=raw_bytes,
                    mask=mask_bytes,
                    prediction=pred_bytes,
                )
                response.samples.append(sample_response)
            except Exception as e:
                print(f"[Error] GetSamples({sid}) failed: {e}")
        return response

    def _apply_zerofy(self, layer, from_ids, to_ids):
        
        in_max = int(layer.incoming_neuron_count)
        out_max = int(layer.neuron_count)

        from_set = {i for i in set(from_ids) if 0 <= i < in_max}
        to_set   = {i for i in set(to_ids)   if 0 <= i < out_max}
        if from_set and to_set and hasattr(layer, "zerofy_connections_from"):
            layer.zerofy_connections_from(from_set, to_set)
        else:
            print("ZEROFY skipped (empty sets or no method)")

    def ManipulateWeights(self, request, context):
        # (f"ExperimentServiceServicer.ManipulateWeights({request})")
        answer = pb2.WeightsOperationResponse(
            success=False, message="Unknown error")
        weight_operations = request.weight_operation

        if weight_operations.op_type == pb2.WeightOperationType.REMOVE_NEURONS:
            layer_id_to_neuron_ids_list = defaultdict(list)
            for neuron_id in weight_operations.neuron_ids:
                layer_id = neuron_id.layer_id
                layer_id_to_neuron_ids_list[layer_id].append(
                    neuron_id.neuron_id)

            for layer_id, neuron_ids in layer_id_to_neuron_ids_list.items():
                experiment.apply_architecture_op(
                    op_type = ArchitectureOpType.PRUNE,
                    layer_id=layer_id,
                    neuron_indices=set(neuron_ids))

            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Pruned {str(dict(layer_id_to_neuron_ids_list))}")
        elif weight_operations.op_type == pb2.WeightOperationType.ADD_NEURONS:
            experiment.apply_architecture_op(
                op_type = ArchitectureOpType.ADD_NEURONS,
                layer_id=weight_operations.layer_id,
                neuron_count=weight_operations.neurons_to_add)
            answer = pb2.WeightsOperationResponse(
                success=True,
                message=\
                    f"Added {weight_operations.neurons_to_add} "
                    f"neurons to layer {weight_operations.layer_id}")
        elif weight_operations.op_type == pb2.WeightOperationType.FREEZE:
            layer_id_to_neuron_ids_list = defaultdict(list)
            for neuron_id in weight_operations.neuron_ids:
                layer_id = neuron_id.layer_id
                layer_id_to_neuron_ids_list[layer_id].append(
                    neuron_id.neuron_id)
            for layer_id, neuron_ids in layer_id_to_neuron_ids_list.items():
                experiment.apply_architecture_op(
                    op_type = ArchitectureOpType.FREEZE,
                    layer_id=layer_id,
                    neuron_ids=neuron_ids)
            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Frozen {str(dict(layer_id_to_neuron_ids_list))}")
        elif weight_operations.op_type == pb2.WeightOperationType.REINITIALIZE:
            for neuron_id in weight_operations.neuron_ids:
                experiment.apply_architecture_op(
                    op_type = ArchitectureOpType.REINITIALIZE,
                    layer_id=neuron_id.layer_id,
                    neuron_indices={neuron_id.neuron_id})

            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Reinitialized {weight_operations.neuron_ids}")
            
        elif weight_operations.op_type == pb2.WeightOperationType.ZEROFY:
            layer_id = weight_operations.layer_id
            layer = experiment.model.get_layer_by_id(layer_id)

            from_ids = list(weight_operations.zerofy_from_incoming_ids)
            to_ids   = list(weight_operations.zerofy_to_neuron_ids)

            if len(weight_operations.zerofy_predicates) > 0:
                frozen = set()
                try:
                    for nid in range(getattr(layer, "neuron_count", 0)):
                        if layer.get_per_neuron_learning_rate(nid) == 0.0:
                            frozen.add(nid)
                except Exception:
                    pass

                older = set()
                try:
                    td_tracker = getattr(layer, "train_dataset_tracker", None)
                    if td_tracker is not None:
                        for nid in range(getattr(layer, "neuron_count", 0)):
                            age = int(td_tracker.get_neuron_age(nid))
                            if age > 0:
                                older.add(nid)
                except Exception:
                    older = set()

                expanded = set(to_ids)
                for p in weight_operations.zerofy_predicates:
                    if p == pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_FROZEN:
                        expanded |= frozen
                    elif p == pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_OLDER:
                        expanded |= older
                to_ids = list(expanded)

            self._apply_zerofy(layer, from_ids, to_ids)

            return pb2.WeightsOperationResponse(
                success=True,
                message=f"Zerofied L{layer_id} from={sorted(set(from_ids))} to={sorted(set(to_ids))}"
            )

        return answer

    def GetWeights(self, request, context):
        # print(f"ExperimentServiceServicer.GetWeights({request})")
        answer = pb2.WeightsResponse(success=True, error_message="")

        neuron_id = request.neuron_id
        layer = None
        try:
            layer = experiment.model.get_layer_by_id(neuron_id.layer_id)
        except Exception as e:
            answer.success = False
            answer.error_messages = str(e)
            return answer

        #answer.neuron_id = request.neuron_id
        answer.neuron_id.CopyFrom(request.neuron_id)
        answer.layer_name = layer.__class__.__name__
        answer.incoming = layer.incoming_neuron_count
        answer.outgoing = layer.neuron_count
        if "Conv2d" in layer.__class__.__name__:
            answer.layer_type = "Conv2d"
            answer.kernel_size = layer.kernel_size[0]
        elif "Linear" in layer.__class__.__name__:
            answer.layer_type = "Linear"

        if neuron_id.neuron_id >= layer.neuron_count:
            answer.success = False
            answer.error_messages = \
                f"Neuron {neuron_id.neuron_id} outside bounds."
            return answer

        if neuron_id.neuron_id < 0:
            # Return all weights
            weights = layer.weight.data.cpu().detach().numpy().flatten()
        else:
            weights = layer.weight[
                neuron_id.neuron_id].data.cpu().detach().numpy().flatten()
        answer.weights.extend(weights)

        return answer
        
    def GetActivations(self, request, context):
        resp = pb2.ActivationResponse(layer_type='conv')
        try:
            ds = experiment.train_loader.dataset if request.origin == "train" else experiment.eval_loader.dataset
        except Exception:
            ds = experiment.train_loader.dataset

        sid = int(request.sample_id)
        if sid < 0 or sid >= len(ds):
            return pb2.ActivationResponse(layer_type="", neurons_count=0)

        x = _get_input_tensor_for_sample(ds, sid)
        layer_id = int(request.layer_id)
        assert experiment.model.get_layer_by_id(layer_id).device == torch.device('cuda')
        intermediaries = {}
        try:
            with torch.no_grad():
                _ = experiment.model.forward(x, intermediary=intermediaries)
        except Exception as e:
            print(f'forward with intermediary failed: {type(e).__name__}: {e}')
        # except TypeError:
        #     try:
        #         with torch.no_grad():
        #             _, intermediaries = experiment.model.forward(x, intermediary_outputs=[layer_id])
        #     except Exception:
        #         captured = {}
        #         layer = experiment.model.get_layer_by_id(layer_id)
        #         def hook(module, inputs, output):
        #             captured['y'] = output.detach().cpu()
        #         handle = layer.register_forward_hook(hook)
        #         try:
        #             with torch.no_grad():
        #                 _ = experiment.model.forward(x)
        #             if 'y' in captured:
        #                 intermediaries[layer_id] = captured['y']
        #         finally:
        #             handle.remove()

        # if layer_id not in intermediaries:
        #     return pb2.ActivationResponse(layer_type="", neurons_count=0)
        # assert experiment.model.get_layer_by_id(layer_id).device == torch.device('cuda')
        # y = intermediaries[layer_id]
        # if hasattr(y, "detach"):
        #     y = y.detach().cpu()
        # y_np = y.numpy()

        # layer = experiment.model.get_layer_by_id(layer_id)
        # layer_type = layer.__class__.__name__
        # resp = pb2.ActivationResponse(layer_type=layer_type)

        # if "Conv2d" in layer_type:
        #     if y_np.ndim != 4:  # (B, C, H, W)
        #         return pb2.ActivationResponse(layer_type=layer_type, neurons_count=0)
        #     _, C, H, W = y_np.shape
        #     resp.neurons_count = int(C)
        #     for c in range(C):
        #         vals = y_np[0, c].astype(np.float32).reshape(-1).tolist()
        #         amap = pb2.ActivationMap(neuron_id=c, values=vals, H=H, W=W)
        #         resp.activations.append(amap)
        # else:
        #     if y_np.ndim == 1:
        #         y_np = y_np.reshape(1, -1)
        #     elif y_np.ndim != 2:
        #         y_np = y_np.reshape(1, -1)

        #     _, N = y_np.shape
        #     resp.neurons_count = int(N)
        #     for n in range(N):
        #         v = float(y_np[0, n])
        #         amap = pb2.ActivationMap(neuron_id=n, values=[v], H=1, W=1)
        #         resp.activations.append(amap)
        # assert experiment.model.get_layer_by_id(layer_id).device == torch.device('cuda')
        return resp


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=6))
    servicer = ExperimentServiceServicer()
    pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    # experiment.toggle_training_status()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
