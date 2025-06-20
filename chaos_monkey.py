import random
import time

# For RPC mode
import grpc
import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc

# For direct mode
try:
    # Try to import from a central location, adjust as per your project layout!
    from weightslab.experiment import ArchitectureOpType
except ImportError:
    ArchitectureOpType = None  # fallback for RPC only runs

# ==== CONFIG ====
MODE = "rpc"  # "rpc" or "direct"
DURATION = 60  # seconds to run
OPERATION_FREQ = (0.05, 0.3)  # random sleep between ops (seconds)
# =================

def chaos_op_rpc(stub):
    # Get available layers from server
    state = stub.ExperimentCommand(pb2.TrainerCommand(get_interactive_layers=True))
    layer_reps = state.layer_representations
    if not layer_reps:
        return

    # Exclude output (last) layer
    output_layer_id = layer_reps[-1].layer_id
    candidate_layers = [layer for layer in layer_reps if layer.layer_id != output_layer_id]
    if not candidate_layers:
        return

    op_type = random.choice([
        pb2.WeightOperationType.ADD_NEURONS,
        pb2.WeightOperationType.REMOVE_NEURONS,
        pb2.WeightOperationType.REINITIALIZE,
        pb2.WeightOperationType.FREEZE
    ])
    layer_info = random.choice(candidate_layers)
    layer_id = layer_info.layer_id
    neurons_to_add = random.randint(1, 4)

    op = None
    if op_type == pb2.WeightOperationType.ADD_NEURONS:
        op = pb2.WeightOperation(
            op_type=op_type,
            layer_id=layer_id,
            neurons_to_add=neurons_to_add
        )
    elif op_type in [pb2.WeightOperationType.REMOVE_NEURONS, pb2.WeightOperationType.REINITIALIZE, pb2.WeightOperationType.FREEZE]:
        if layer_info.neurons_count == 0:
            return
        neuron_ids = [pb2.NeuronId(layer_id=layer_id, neuron_id=random.randint(0, layer_info.neurons_count-1))]
        op = pb2.WeightOperation(
            op_type=op_type,
            neuron_ids=neuron_ids
        )

    if op is not None:
        req = pb2.WeightsOperationRequest(weight_operation=op)
        response = stub.ManipulateWeights(req)
        print(f"[RPC][Chaos] Sent {op.op_type} on layer {layer_id}: {response.message}")

def chaos_op_direct(experiment):
    if ArchitectureOpType is None:
        raise ImportError("ArchitectureOpType enum could not be imported.")

    model = experiment.model
    layers = model.layers
    if not layers or len(layers) < 2:
        return
    # Exclude output layer
    non_output_layers = layers[:-1]
    layer = random.choice(non_output_layers)
    op_type = random.choice([
        ArchitectureOpType.ADD_NEURONS,
        ArchitectureOpType.PRUNE,
        ArchitectureOpType.REINIT,
        ArchitectureOpType.FREEZE,
    ])
    n = random.randint(1, 3)
    if layer.neuron_count == 0:
        return
    neuron_indices = set(random.sample(range(layer.neuron_count), min(n, layer.neuron_count)))
    print(f"[Direct][Chaos] {op_type} on layer {layer.get_module_id()}, neurons: {neuron_indices}")

    with experiment.architecture_guard:
        if op_type == ArchitectureOpType.ADD_NEURONS:
            model.add_neurons(layer_id=layer.get_module_id(), neuron_count=n)
        elif op_type == ArchitectureOpType.PRUNE:
            model.prune(layer_id=layer.get_module_id(), neuron_indices=neuron_indices)
        elif op_type == ArchitectureOpType.REINITIALIZE:
            model.reinit_neurons(layer_id=layer.get_module_id(), neuron_indices=neuron_indices)
        elif op_type == ArchitectureOpType.FREEZE:
            model.freeze(layer_id=layer.get_module_id(), neuron_ids=neuron_indices)

def run_chaos(mode="rpc", duration=60):
    t0 = time.time()
    if mode == "rpc":
        channel = grpc.insecure_channel('localhost:50051')
        stub = pb2_grpc.ExperimentServiceStub(channel)
        op_func = lambda: chaos_op_rpc(stub)
    else:
        from trainer_worker import experiment  # Must be same process!
        op_func = lambda: chaos_op_direct(experiment)
    while time.time() - t0 < duration:
        try:
            op_func()
        except Exception as e:
            print("[Chaos] Exception:", e)
        time.sleep(random.uniform(*OPERATION_FREQ))

if __name__ == "__main__":
    print("Starting Chaos Monkey!")
    run_chaos(mode=MODE, duration=DURATION)
