import pdb
from typing import List, Set, Dict
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets as ds

from weightslab.experiment import Experiment

from weightslab.model_with_ops import NetworkWithOps
from weightslab.model_with_ops import DepType
from weightslab.modules_with_ops import Conv2dWithNeuronOps
from weightslab.modules_with_ops import LinearWithNeuronOps
from weightslab.modules_with_ops import BatchNorm2dWithNeuronOps

from weightslab.tracking import TrackingMode
from weightslab.tracking import add_tracked_attrs_to_input_tensor


import torch.fx as fx


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dWithNeuronOps(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2dWithNeuronOps(out_channels)
        self.conv2 = Conv2dWithNeuronOps(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2dWithNeuronOps(out_channels)
        self.downsample = downsample

    def children(self):
        return [self.conv1, self.bn1, self.conv2, self.bn2]

    # def define_deps(self):
    #     self.register_dependencies([
    #         (self.conv1, self.bn1, DepType.SAME),
    #         (self.bn1, self.conv2, DepType.INCOMING),
    #         (self.conv2, self.bn2, DepType.SAME),
    #     ])
    #     self.flatten_conv_id = None
    
    def define_dependencies(self, dep_manager):
        dep_manager.register_dependencies([
            (self.conv1, self.bn1, DepType.SAME),
            (self.bn1, self.conv2, DepType.INCOMING),
            (self.conv2, self.bn2, DepType.SAME),
        ])

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


class ResNet34(NetworkWithOps):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED

        self.conv1 = Conv2dWithNeuronOps(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2dWithNeuronOps(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearWithNeuronOps(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                Conv2dWithNeuronOps(in_channels, out_channels, kernel_size=1, stride=stride),
                BatchNorm2dWithNeuronOps(out_channels)
            )
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def children(self):
        lst = [self.conv1, self.bn1]
        for block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            lst.extend(block.children())
        lst.append(self.fc)
        return lst

    def define_deps(self):
        
        for sequential in [self.layer1, self.layer2, self.layer3, self.layer4]:
            last_block = None
            for block in sequential.children():
                block.define_dependencies(self.dep_manager)
                if last_block is not None:
                    self.dep_manager.register_dependencies([
                        (last_block.bn2, block.conv1, DepType.INCOMING)
                    ])
            
        self.register_dependencies([
            (self.conv1, self.bn1, DepType.SAME),
            # (self.bn1, list(self.layer1.children())[0], DepType.INCOMING),
            # (list(self.layer1.children())[-1], list(self.layer2.children())[0], DepType.INCOMING),
            # (list(self.layer2.children())[-1], list(self.layer3.children())[0], DepType.INCOMING),
            # (list(self.layer3.children())[-1], list(self.layer4.children())[0], DepType.INCOMING),
            # (list(self.layer4.children())[-1], self.fc, DepType.INCOMING),
        ])
        
        # for sequential in [self.layer1, self.layer2, self.layer3, self.layer4]:
        #     for block in sequential.children():
        #         for layer_w_ops in block.children():
        #             print("layer_w_ops : ", str(layer_w_ops)[:20])
                    
                    
            
    


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


train_set = ds.FashionMNIST(
    "./data", download=True, transform=T.Compose([T.ToTensor()]))
test_set = ds.FashionMNIST(
    "./data", download=True, train=False, transform=T.Compose([T.ToTensor()]))  

device = th.device("cuda:0")
# device = th.device("cpu")

def get_exp():
    model = ResNet34()

    # traced_model = fx.symbolic_trace(model)
    # print(traced_model.graph)

    traced_model = fx.symbolic_trace(model)
    
    

    # Access nodes in the graph
    for node in traced_model.graph.nodes:
        print(f"Node: {node.name}, Target: {node.target}, Args: {node.args}, Kwargs: {node.kwargs}")
    
    
    # print("model", model)
    # for child_idx, child in enumerate(model.children()):
    #     print(child_idx, ": " , type(child))
    # model.define_deps()
    # exp = Experiment(
    #     model=model, optimizer_class=optim.Adam,
    #     train_dataset=train_set,
    #     eval_dataset=test_set,
    #     device=device, learning_rate=1e-3, batch_size=100,
    #     name="v0",
    #     root_log_dir='fmnist-resnet34',
    #     logger=Dash("fmnist-resnet34"),
    #     skip_loading=False)

    # def stateful_difference_monitor_callback():
    #     exp.display_stats()

    # exp.register_train_loop_callback(stateful_difference_monitor_callback)

    # return exp

# import pdb; pdb.set_trace()
exo = get_exp()