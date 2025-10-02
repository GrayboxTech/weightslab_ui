# WeightsLab UI

WeightsLab is a powerful tool for editing and inspecting AI model weights during training.
This early prototype helps you debug and fix common training issues through interactive weight manipulation and granular analysis.

## What Problems Does It Solve?
WeightsLab addresses critical training challenges:

* Overfitting and training plateaus
* Dataset insights and optimization
* Over/Under parameterization

## Key Capabilities
The granular statistics and interactive paradigm enables powerful workflows:

* Monitor granular insights on data samples and weights parameters
* Discard low quality samples by click or query
* Create slices of data and discard them during training
* Iterative pruning or growing of the architectures by click or query

## Getting Started
### Installation

Clone and install the framework:

```
git clone https://github.com/GrayboxTech/weightslab.git
cd weightslab
pip install -e .
```

Clone the UI repository:
```
git clone git@github.com:GrayboxTech/weightslab_ui.git
cd weightslab_ui
```

Compile RPC messages:
```
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. experiment_service.proto
```

## Running WeightsLab

### Define your experiment

```
# your_custom_exp.py

import torch as th
...

from weightslab.experiment import Experiment
from weightslab.model_with_ops import *
from weightslab.modules_with_ops import *
from weightslab.tracking import *


class Model(NetworkWithOps, nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED

        self.conv1 = Conv2dWithNeuronOps(1, 128, kernel_size=3, padding=1)
        self.bnrm1 = BatchNorm2dWithNeuronOps(128)
        ...

        self.define_deps()

    def children(self):
        return [self.conv1, self.bnrm1,]

    def define_deps(self):
        self.register_dependencies([
            (self.conv1, self.bnrm1, DepType.SAME),
            (self.bnrm1,   ... , DepType.INCOMING),
            ...
        ])
        self.flatten_conv_id = ...

    def forward(self, x, intermediary_outputs=None):
        self.maybe_update_age(x)
        ...

class CustomOrStandard(Dataset):
    def __init__(self, folders, image_size=None):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...


def get_exp():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = Model(...)
    train_dataset = CustomOrStandard(...)
    eval_dataset = CustomOrStandard(...)
    metrics = {"acc": BinaryAccuracy().to(device),} 

    exp = Experiment(
        model=model,
        optimizer_class=optim.Adam,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        device=device,
        learning_rate=1e-3,
        batch_size=32,
        training_steps_to_do=1000,
        name="experiment0",
        metrics=metrics,
        root_log_dir="task",
        logger=Dash("task"),
        criterion=nn.L1Loss(reduction="mean")
    )
    return exp
```
### Load your experiment in the trainer_worker
```
...

from your_custom_exp import get_exp
...
```

### Start the trainer process:

```
python trainer_worker.py
```

## Launch the UI monitoring process:
```
python weights_lab.py --root_directory=task
```

### Open the provided URL (typically http://127.0.0.1:8050/)
