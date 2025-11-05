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
import traceback

# 🔽 MINIMAL CHANGE: point to your experiment + expose IM_MEAN/IM_STD
from mnist_exp import get_exp, IM_MEAN, IM_STD

experiment = get_exp()

# optional: start training immediately (otherwise start from UI)
# experiment.set_is_training(True)

def training_thread_callback():
    while True:
        if experiment.get_is_training():
            experiment.train_step_or_eval_full()

training_thread = Thread(target=training_thread_callback)
training_thread.start()

samples_packaging_timer = ScopeTimer("samples_packaging")

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
        lr = get_layer_representation(layer)
        if lr is not None:
            layer_representations.append(lr)
    return layer_representations

def make_task_field(name, value):
    if isinstance(value, float):
        return pb2.TaskField(name=name, float_value=value)
    elif isinstance(value, bool):
        return pb2.TaskField(name=name, bool_value=value)
    elif isinstance(value, int):
        return pb2.TaskField(name=name, int_value=value)
    elif isinstance(value, str):
        return pb2.TaskField(name=name, string_value=value)
    elif isinstance(value, bytes):
        return pb2.TaskField(name=name, bytes_value=value)
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

def _class_ids(x, num_classes=None, ignore_index=255):
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 2:
        return []
    u = np.unique(x.astype(np.int64))
    u = u[u != int(ignore_index)]
    if num_classes is not None:
        u = u[(u >= 0) & (u < int(num_classes))]
    return [int(v) for v in u.tolist()]

def _labels_from_mask_path_histogram(path, num_classes=None, ignore_index=255):
    with Image.open(path) as im:
        if im.mode not in ("P", "L"):
            im = im.convert("L")
        hist = im.histogram()  # length 256
    ub = 256 if num_classes is None else int(num_classes)
    ids = [i for i, cnt in enumerate(hist[:ub]) if cnt > 0]
    if ignore_index is not None:
        ig = int(ignore_index)
        ids = [i for i in ids if i != ig]
    return ids

def get_data_set_representation(dataset) -> pb2.SampleStatistics:
    sample_stats = pb2.SampleStatistics()
    sample_stats.sample_count = len(dataset.wrapped_dataset)

    tasks = getattr(experiment, "tasks", None)
    is_multi_task = bool(tasks) and len(tasks) > 1
    sample_stats.task_type = "multi-task" if is_multi_task else getattr(
        experiment, "task_type", getattr(dataset, "task_type", "classification")
    )

    ignore_index = getattr(dataset, "ignore_index", 255)
    num_classes  = getattr(dataset, "num_classes", getattr(experiment, "num_classes", None))

    for sample_id, row in enumerate(dataset.as_records()):
        record = pb2.RecordMetadata(
            sample_id=row.get('sample_id', sample_id),
            sample_last_loss=float(row.get('prediction_loss', -1)),
            sample_encounters=int(row.get('encountered', row.get('exposure_amount', 0))),
            sample_discarded=bool(row.get('deny_listed', False)),
            task_type=sample_stats.task_type,
        )

        if is_multi_task:
            preview_attached = False
            for t in tasks:
                loss_key, pred_key = f"loss/{t.name}", f"pred/{t.name}"
                if loss_key in row:
                    record.extra_fields.append(make_task_field(loss_key, float(np.array(row[loss_key]).reshape(-1)[0])))
                if pred_key in row:
                    arr = np.array(row[pred_key].detach().cpu() if hasattr(row[pred_key], "detach") else row[pred_key])
                    if arr.ndim >= 2 and not preview_attached:
                        record.prediction_raw = tensor_to_bytes(row[pred_key], mean=IM_MEAN, std=IM_STD)
                        preview_attached = True
                    elif arr.size == 1:
                        record.extra_fields.append(make_task_field(pred_key, float(arr.reshape(-1)[0])))
            target = row.get("target", -1)
            record.sample_label.extend([int(target)])
        else:
            task_type = sample_stats.task_type
            if task_type == "segmentation":
                label = row.get("target")
                if isinstance(label, str):
                    target_list = _labels_from_mask_path_histogram(label, num_classes, ignore_index)
                else:
                    target_list = _class_ids(label, num_classes, ignore_index)
                pred_list = _class_ids(row.get("prediction_raw"), num_classes, ignore_index)
            else:
                target = row.get("label", row.get("target", -1))
                pred   = row.get("prediction_raw", -1)
                target_list = [int(target)] if not isinstance(target, (list, np.ndarray)) else [int(np.array(target).item())]
                pred_list   = [int(pred)]   if not isinstance(pred, (list, np.ndarray))   else [int(np.array(pred).item())]
            record.sample_label.extend(target_list)
            record.sample_prediction.extend(pred_list)

        sample_stats.records.append(record)
    return sample_stats

def _maybe_denorm(img_t, mean=None, std=None):
    if mean is None or std is None:
        return img_t
    if img_t.ndim != 3 or img_t.shape[0] not in (1,3):
        return img_t
    m = torch.tensor(mean, dtype=img_t.dtype, device=img_t.device).view(-1,1,1)
    s = torch.tensor(std,  dtype=img_t.dtype, device=img_t.device).view(-1,1,1)
    return img_t * s + m

def tensor_to_bytes(tensor, mean=None, std=None):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    if tensor.dtype.is_floating_point:
        tensor = _maybe_denorm(tensor, mean, std)
        tensor = torch.clamp(tensor, 0.0, 1.0)
    if tensor.ndim == 3 and tensor.shape[0] > 1:
        np_img = (tensor.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        mode = "RGB"
    else:
        np_img = tensor.squeeze(0).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        mode = "L"
    img = Image.fromarray(np_img, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format='jpeg', quality=85)
    return buf.getvalue()

def load_raw_image(dataset, index):
    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    if hasattr(wrapped, "images") and isinstance(wrapped.images, list):
        img_path = wrapped.images[index]
        img = Image.open(img_path)
        return img.convert("RGB")
    elif hasattr(wrapped, "files") and isinstance(wrapped.files, list):
        img_path = wrapped.files[index]
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

def _get_input_tensor_for_sample(dataset, sample_id, device):
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
    tensor = tensor.to(device)
    return tensor

def process_sample(sid, dataset, do_resize, resize_dims):
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
        if do_resize:
            pil_img = pil_img.resize(resize_dims, Image.BILINEAR)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        transformed_bytes = buf.getvalue()

        try:
            raw = load_raw_image(dataset, sid)
            if do_resize:
                raw = raw.resize(resize_dims, Image.BILINEAR)
            raw_buf = io.BytesIO()
            raw.save(raw_buf, format='PNG')
            raw_bytes = raw_buf.getvalue()
        except Exception:
            raw_bytes = transformed_bytes

        tasks = getattr(experiment, "tasks", None)
        is_multi_task = bool(tasks) and len(tasks) > 1
        task_type = getattr(experiment, "task_type", getattr(dataset, "task_type", "classification"))
        if is_multi_task:
            task_type = "multi-task"

        cls_label = -1
        mask_bytes = b""
        pred_bytes = b""

        if task_type == "classification":
            if isinstance(label, (list, np.ndarray)):
                cls_label = int(np.array(label).item())
            elif hasattr(label, 'cpu'):
                cls_label = int(np.array(label.cpu()).item())
            else:
                cls_label = int(label)
        elif task_type == "segmentation":
            num_classes = getattr(dataset, "num_classes", 21)
            try:
                mask_bytes = mask_to_png_bytes(label, num_classes=num_classes)
            except Exception:
                mask_bytes = mask_to_png_bytes(label)
            try:
                if hasattr(dataset, "get_prediction_mask"):
                    pred_mask = dataset.get_prediction_mask(sid)
                    if pred_mask is not None:
                        pred_bytes = mask_to_png_bytes(pred_mask, num_classes=num_classes)
            except Exception:
                pred_bytes = b""
        elif task_type == "reconstruction":
            mask_bytes = raw_bytes if raw_bytes else transformed_bytes
            try:
                if hasattr(dataset, "get_prediction_mask"):
                    recon = dataset.get_prediction_mask(sid, task_name='recon')
                    if recon is not None:
                        r = recon.detach().cpu() if isinstance(recon, torch.Tensor) else torch.tensor(recon)
                        if r.ndim == 2:
                            r = r.unsqueeze(0)
                        pred_bytes = tensor_to_bytes(r, mean=IM_MEAN, std=IM_STD)
            except Exception:
                pred_bytes = b""
        elif task_type == "multi-task":
            if isinstance(label, (list, np.ndarray)):
                cls_label = int(np.array(label).item())
            elif hasattr(label, 'cpu'):
                cls_label = int(np.array(label.cpu()).item())
            else:
                cls_label = int(label)
            try:
                if hasattr(dataset, "get_prediction_mask"):
                    recon = dataset.get_prediction_mask(sid, task_name='recon')
                    if recon is not None:
                        r = recon.detach().cpu() if isinstance(recon, torch.Tensor) else torch.tensor(recon)
                        if r.ndim == 2:
                            r = r.unsqueeze(0)
                        pred_bytes = tensor_to_bytes(r, mean=IM_MEAN, std=IM_STD)
            except Exception:
                pred_bytes = b""

        return sid, transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes

    except Exception as e:
        print(f"[Error] GetSamples({sid}) failed: {e}")
        return sid, None, None, -1, b"", b""

class ExperimentServiceServicer(pb2_grpc.ExperimentServiceServicer):
    def StreamStatus(self, request_iterator, context):
        global experiment
        while True:
            log = experiment.logger.queue.get()
            if log is None:
                break
            metrics_status, annotat_status = None, None
            if "metric_name" in log:
                metrics_status = pb2.MetricsStatus(
                    name=log["metric_name"],
                    value=log["metric_value"],
                )
            elif "annotation" in log:
                annotat_status = pb2.AnnotatStatus(name=log["annotation"])
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
        if request.HasField('hyper_parameter_change'):
            hyper_parameters = request.hyper_parameter_change.hyper_parameters

            if hyper_parameters.HasField('is_training'):
                experiment.set_is_training(hyper_parameters.is_training)
            if hyper_parameters.HasField('learning_rate'):
                experiment.set_learning_rate(hyper_parameters.learning_rate)
            if hyper_parameters.HasField('batch_size'):
                experiment.set_batch_size(hyper_parameters.batch_size)
            if hyper_parameters.HasField('training_steps_to_do'):
                experiment.set_training_steps_to_do(hyper_parameters.training_steps_to_do)
            if hyper_parameters.HasField('experiment_name'):
                experiment.set_name(hyper_parameters.experiment_name)

            return pb2.CommandResponse(success=True, message="Hyper parameter changed")

        if request.HasField('deny_samples_operation'):
            denied_cnt = len(request.deny_samples_operation.sample_ids)
            experiment.train_loader.dataset.denylist_samples(
                set(request.deny_samples_operation.sample_ids),
                accumulate=request.deny_samples_operation.accumulate
            )
            return pb2.CommandResponse(success=True, message=f"Denied {denied_cnt} train samples")

        if request.HasField('deny_eval_samples_operation'):
            denied_cnt = len(request.deny_eval_samples_operation.sample_ids)
            experiment.eval_loader.dataset.denylist_samples(
                set(request.deny_eval_samples_operation.sample_ids),
                accumulate=request.deny_eval_samples_operation.accumulate
            )
            return pb2.CommandResponse(success=True, message=f"Denied {denied_cnt} eval samples")

        if request.HasField('remove_from_denylist_operation'):
            allowed = set(request.remove_from_denylist_operation.sample_ids)
            experiment.train_loader.dataset.allowlist_samples(allowed)
            return pb2.CommandResponse(success=True, message=f"Un-denied {len(allowed)} train samples")

        if request.HasField('remove_eval_from_denylist_operation'):
            allowed = set(request.remove_eval_from_denylist_operation.sample_ids)
            experiment.eval_loader.dataset.allowlist_samples(allowed)
            return pb2.CommandResponse(success=True, message=f"Un-denied {len(allowed)} eval samples")

        if request.HasField('load_checkpoint_operation'):
            checkpoint_id = request.load_checkpoint_operation.checkpoint_id
            experiment.load(checkpoint_id)

        response = pb2.CommandResponse(success=True, message="")
        if request.get_hyper_parameters:
            response.hyper_parameters_descs.extend(get_hyper_parameters_pb(HYPER_PARAMETERS))
        if request.get_interactive_layers:
            if request.HasField('get_single_layer_info_id'):
                response.layer_representations.extend([
                    get_layer_representation(experiment.model.get_layer_by_id(
                        request.get_single_layer_info_id))])
            else:
                response.layer_representations.extend(get_layer_representations(experiment.model))
        if request.get_data_records:
            if request.get_data_records == "train":
                response.sample_statistics.CopyFrom(get_data_set_representation(experiment.train_loader.dataset))
                response.sample_statistics.origin = "train"
            elif request.get_data_records == "eval":
                response.sample_statistics.CopyFrom(get_data_set_representation(experiment.eval_loader.dataset))
                response.sample_statistics.origin = "eval"
        return response

    def GetSample(self, request, context):
        if not request.HasField('sample_id') or not request.HasField('origin'):
            return pb2.SampleRequestResponse(error_message="Invalid request. Provide sample_id & origin.")
        if request.origin not in ["train", "eval"]:
            return pb2.SampleRequestResponse(error_message=f"Invalid origin {request.origin}")
        if request.sample_id < 0:
            return pb2.SampleRequestResponse(error_message=f"Invalid sample_id {request.sample_id}")

        dataset = experiment.train_loader.dataset if request.origin == "train" else experiment.eval_loader.dataset
        if request.sample_id >= len(dataset):
            return pb2.SampleRequestResponse(error_message=f"Sample {request.sample_id} not found.")

        transformed_tensor, idx, label = dataset._getitem_raw(request.sample_id)
        transformed_image_bytes = tensor_to_bytes(transformed_tensor)

        try:
            pil_img = load_raw_image(dataset, request.sample_id)
            buf = io.BytesIO()
            pil_img.save(buf, format='jpeg', quality=85)
            raw_image_bytes = buf.getvalue()
        except Exception as e:
            return pb2.SampleRequestResponse(error_message=str(e))

        return pb2.SampleRequestResponse(
            sample_id=request.sample_id,
            origin=request.origin,
            label=label,
            raw_data=raw_image_bytes,
            data=transformed_image_bytes,
        )

    def GetSamples(self, request, context):
        print(f"ExperimentServiceServicer.GetSamples({request})")
        import concurrent.futures
        dataset = experiment.train_loader.dataset if request.origin == "train" else experiment.eval_loader.dataset
        response = pb2.BatchSampleResponse()

        do_resize = request.HasField("resize_width") and request.HasField("resize_height")
        resize_dims = (request.resize_width, request.resize_height) if do_resize else None
        task_type = getattr(dataset, "task_type", getattr(experiment, "task_type", "classification"))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut_map = {executor.submit(process_sample, sid, dataset, do_resize, resize_dims): sid
                       for sid in request.sample_ids}
            results = {}
            for future in concurrent.futures.as_completed(fut_map):
                sid, transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes = future.result()
                results[sid] = (transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes)

        for sid in request.sample_ids:
            transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes = results.get(
                sid, (None, None, -1, b"", b""))
            if transformed_bytes is None or raw_bytes is None:
                continue

            if task_type == "segmentation":
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid, label=cls_label, data=transformed_bytes, raw_data=raw_bytes,
                    mask=mask_bytes, prediction=pred_bytes or b"")
            elif pred_bytes and len(pred_bytes) > 0:
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid, label=-1, data=transformed_bytes, raw_data=raw_bytes,
                    mask=b"", prediction=pred_bytes)
            else:
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid, label=cls_label, data=transformed_bytes, raw_data=raw_bytes,
                    mask=b"", prediction=b"")
            response.samples.append(sample_response)
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
        answer = pb2.WeightsOperationResponse(success=False, message="Unknown error")
        weight_operations = request.weight_operation

        if weight_operations.op_type == pb2.WeightOperationType.REMOVE_NEURONS:
            layer_id_to_neuron_ids_list = defaultdict(list)
            for neuron_id in weight_operations.neuron_ids:
                layer_id_to_neuron_ids_list[neuron_id.layer_id].append(neuron_id.neuron_id)
            for layer_id, neuron_ids in layer_id_to_neuron_ids_list.items():
                experiment.apply_architecture_op(
                    op_type=ArchitectureOpType.PRUNE, layer_id=layer_id, neuron_indices=set(neuron_ids))
            answer = pb2.WeightsOperationResponse(success=True, message=f"Pruned {str(dict(layer_id_to_neuron_ids_list))}")

        elif weight_operations.op_type == pb2.WeightOperationType.ADD_NEURONS:
            experiment.apply_architecture_op(
                op_type=ArchitectureOpType.ADD_NEURONS,
                layer_id=weight_operations.layer_id,
                neuron_count=weight_operations.neurons_to_add)
            answer = pb2.WeightsOperationResponse(
                success=True, message=f"Added {weight_operations.neurons_to_add} neurons to layer {weight_operations.layer_id}")

        elif weight_operations.op_type == pb2.WeightOperationType.FREEZE:
            layer_id_to_neuron_ids_list = defaultdict(list)
            for neuron_id in weight_operations.neuron_ids:
                layer_id_to_neuron_ids_list[neuron_id.layer_id].append(neuron_id.neuron_id)
            for layer_id, neuron_ids in layer_id_to_neuron_ids_list.items():
                experiment.apply_architecture_op(
                    op_type=ArchitectureOpType.FREEZE, layer_id=layer_id, neuron_ids=neuron_ids)
            answer = pb2.WeightsOperationResponse(success=True, message=f"Frozen {str(dict(layer_id_to_neuron_ids_list))}")

        elif weight_operations.op_type == pb2.WeightOperationType.REINITIALIZE:
            for neuron_id in weight_operations.neuron_ids:
                experiment.apply_architecture_op(
                    op_type=ArchitectureOpType.REINITIALIZE, layer_id=neuron_id.layer_id, neuron_indices={neuron_id.neuron_id})
            answer = pb2.WeightsOperationResponse(success=True, message=f"Reinitialized {weight_operations.neuron_ids}")

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
        answer = pb2.WeightsResponse(success=True, error_message="")

        neuron_id = request.neuron_id
        try:
            layer = experiment.model.get_layer_by_id(neuron_id.layer_id)
        except Exception as e:
            answer.success = False
            answer.error_messages = str(e)
            return answer

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
            answer.error_messages = f"Neuron {neuron_id.neuron_id} outside bounds."
            return answer

        if neuron_id.neuron_id < 0:
            weights = layer.weight.data.cpu().detach().numpy().flatten()
        else:
            weights = layer.weight[neuron_id.neuron_id].data.cpu().detach().numpy().flatten()
        answer.weights.extend(weights)
        return answer

    def GetActivations(self, request, context):
        print(f"ExperimentServiceServicer.GetActivations({request})")
        empty_resp = pb2.ActivationResponse(layer_type="", neurons_count=0)
        try:
            last_layer = experiment.model.layers[-1]
            last_layer_id = int(last_layer.get_module_id())
            if int(request.layer_id) == last_layer_id:
                return empty_resp

            ds = experiment.train_loader.dataset if request.origin != "eval" \
                 else experiment.eval_loader.dataset

            sid = int(request.sample_id)
            if request.sample_id < 0 or request.sample_id >= len(ds):
                raise ValueError(f"No sample id {request.sample_id} for {request.origin}")

            x = _get_input_tensor_for_sample(ds, request.sample_id, experiment.device)
            with torch.no_grad():
                intermediaries = {request.layer_id: None}
                _ = experiment.model.forward(x, intermediary_outputs=intermediaries)

            if intermediaries[request.layer_id] is None:
                raise ValueError(f"No intermediary layer {request.layer_id}")

            layer = experiment.model.get_layer_by_id(request.layer_id)
            layer_type = layer.__class__.__name__
            amap = intermediaries[request.layer_id].squeeze(0).detach().cpu().numpy()
            resp = pb2.ActivationResponse(layer_type=layer_type)

            C, H, W = 1, 1, 1
            if amap.ndim == 3:  # Conv2d output (C, H, W)
                C, H, W = amap.shape
            elif amap.ndim == 1:  # Linear output (C,)
                C = amap.shape
            resp.neurons_count = C
            for c in range(C):
                vals = amap[c].astype(np.float32).reshape(-1).tolist()
                if not isinstance(vals, list):
                    vals = [vals]
                resp.activations.append(pb2.ActivationMap(neuron_id=c, values=vals, H=H, W=W))
            return resp
        except (ValueError, Exception) as e:
            print(f"Error in GetActivations: {str(e)}", f"Traceback: {traceback.format_exc()}")
        return empty_resp

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=6))
    servicer = ExperimentServiceServicer()
    pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
