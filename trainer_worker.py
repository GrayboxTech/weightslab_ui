import grpc
import time
import io

import numpy as np

from concurrent import futures
from threading import Thread
from PIL import Image
from typing import List, Tuple, Iterable

import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from collections import defaultdict
import pickle

from scope_timer import ScopeTimer

from fashion_mnist_exp import get_exp

# from hct_kaggle_exp import get_exp
# from cifar_exp import get_exp
# from imagenet_exp import get_exp
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
            print(f"[TRAIN] Steps left: {experiment.get_training_steps_to_do()}")


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


def get_data_set_representation(dataset) -> pb2.SampleStatistics:
    # print("[BACKEND].get_data_set_representation")

    all_rows = list(dataset.as_records())

    sample_stats = pb2.SampleStatistics()
    sample_stats.origin = "train"
    sample_stats.sample_count = len(dataset.wrapped_dataset)

    task_type = getattr(dataset, "task_type", "classification")

    for sample_id, row in enumerate(all_rows):
        label = row.get("label") or row.get("target")
        prediction_raw = row.get("prediction_raw", None)

        mask_preview = None
        if task_type == "segmentation" and prediction_raw is not None:
            if hasattr(prediction_raw, "cpu"):
                arr = prediction_raw.cpu().numpy()
            else:
                arr = np.array(prediction_raw)
            if arr.ndim == 3 and arr.shape[0] == 1:  # (1, H, W)
                arr = arr.squeeze(0)
            try:
                from PIL import Image
                img = Image.fromarray(arr.astype(np.uint8))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                mask_preview = buf.getvalue()
            except Exception as e:
                print(f"Could not generate mask preview: {e}")

        prediction_raw_bytes = None
        if prediction_raw is not None:
            try:
                prediction_raw_bytes = pickle.dumps(prediction_raw)
            except Exception as e:
                print(f"Could not serialize prediction_raw: {e}")

        record = pb2.RecordMetadata(
            sample_id = row.get('sample_id', sample_id),
            sample_label = int(label) if label is not None else -1,
            sample_last_loss = row.get('prediction_loss', -1),
            sample_encounters = row.get('encountered', row.get('exposure_amount', 0)),
            sample_discarded = row.get('deny_listed', False),
            mask_preview = mask_preview if mask_preview is not None else b"",
            prediction_raw = prediction_raw_bytes if prediction_raw_bytes is not None else b"",
            task_type = task_type,
        )

        core_fields = {
            'label', 'target', 'prediction_raw', 'prediction_loss', 'exposure_amount',
            'deny_listed', 'sample_id', 'encountered'
        }
        for k, v in row.items():
            if k not in core_fields:
                try:
                    record.extra_fields.append(make_task_field(k, v))
                except Exception as e:
                    print(f"[EXTRA FIELD ERROR] Could not add extra field '{k}': {e}")

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
    img.save(buf, format='png')
    return buf.getvalue()

def load_raw_image(dataset, index):
    wrapped = dataset.wrapped_dataset if hasattr(dataset, "wrapped_dataset") else dataset

    if hasattr(wrapped, "data"):
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


class ExperimentServiceServicer(pb2_grpc.ExperimentServiceServicer):
    def StreamStatus(self, request_iterator, context):
        global experiment
        while True:
            log = experiment.logger.queue.get()
            if "metric_name" in log:
                print(f"[LOG] {log['metric_name']} = {log['metric_value']:.2f}")

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
                set(request.deny_samples_operation.sample_ids))
            return pb2.CommandResponse(
                success=True, message=f"Denied {denied_cnt} train samples")
        if request.HasField('deny_eval_samples_operation'):
            denied_cnt = len(request.deny_eval_samples_operation.sample_ids)
            experiment.eval_loader.dataset.denylist_samples(
                set(request.deny_eval_samples_operation.sample_ids))
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
            pil_img.save(buf, format='PNG')
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

        for sid in request.sample_ids:
            try:
                tensor, _, label = dataset._getitem_raw(sid)
                transformed = tensor_to_bytes(tensor)
                raw = load_raw_image(dataset, sid)
                buf = io.BytesIO()
                raw.save(buf, format="PNG")
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    origin=request.origin,
                    label=label,
                    data=transformed,
                    raw_data=buf.getvalue()
                )
                response.samples.append(sample_response)
            except Exception as e:
                print(f"[Error] GetSamples({sid}) failed: {e}")
        return response


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
                experiment.model.prune(
                    layer_id=layer_id,
                    neuron_indices=set(neuron_ids))

            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Pruned {str(dict(layer_id_to_neuron_ids_list))}")
        elif weight_operations.op_type == pb2.WeightOperationType.ADD_NEURONS:
            experiment.model.add_neurons(
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
                experiment.model.freeze(
                    layer_id=layer_id,
                    neuron_ids=neuron_ids)
            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Frozen {str(dict(layer_id_to_neuron_ids_list))}")
        elif weight_operations.op_type == pb2.WeightOperationType.REINITIALIZE:
            for neuron_id in weight_operations.neuron_ids:
                experiment.model.reinit_neurons(
                    layer_id=neuron_id.layer_id,
                    neuron_indices={neuron_id.neuron_id})

            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Reinitialized {weight_operations.neuron_ids}")
            
        elif weight_operations.op_type == pb2.WeightOperationType.ZEROFY:
            experiment.model.zerofy(layer_id=weight_operations.layer_id)
            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Zerofy applied to new neurons in layer {weight_operations.layer_id}")

        return answer

    def GetWeights(self, request, context):
        # print(f"ExperimentServiceServicer.GetWeights({request})")
        answer = pb2.WeightsResponse(success=True, error_messages="")

        neuron_id = request.neuron_id
        layer = None
        try:
            layer = experiment.model.get_layer_by_id(neuron_id.layer_id)
        except Exception as e:
            answer.success = False
            answer.error_messages = str(e)
            return answer

        answer.neuron_id = request.neuron_id
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
            weights = layer.weights.data.cpu().detach().numpy().flatten()
        else:
            weights = layer.weights[
                neuron_id.neuron_id].data.cpu().detach().numpy().flatten()
        answer.weights.extend(weights)

        return answer

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
