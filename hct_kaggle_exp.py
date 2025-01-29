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

from torch.utils.data import Dataset

from board import Dash


# class FashionCNN(NetworkWithOps):
#     def __init__(self):
#         super(FashionCNN, self).__init__()
#         self.tracking_mode = TrackingMode.DISABLED

#         self.layer1 = Conv2dWithNeuronOps(
#             in_channels=1, out_channels=8, kernel_size=3, padding=1)
#         self.bnorm1 = BatchNorm2dWithNeuronOps(8)
#         self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.layer2 = Conv2dWithNeuronOps(
#             in_channels=8, out_channels=8, kernel_size=3)
#         self.bnorm2 = BatchNorm2dWithNeuronOps(8)
#         self.mpool2 = nn.MaxPool2d(2)

#         self.fc1 = LinearWithNeuronOps(in_features=8*6*6, out_features=10)
#         # self.bnorm3 = BatchNorm2dWithNeuronOps(600)
#         self.drop1 = nn.Dropout(0.25)
#         self.fc2 = LinearWithNeuronOps(in_features=10, out_features=10)
#         # self.bnorm4 = BatchNorm2dWithNeuronOps(120)
#         self.fc3 = LinearWithNeuronOps(in_features=10, out_features=10)
#         self.softmax = nn.Softmax(dim=1)

#     def children(self):
#         return [
#             self.layer1, self.bnorm1, self.layer2, self.bnorm2,
#             self.fc1, self.fc2, self.fc3
#         ]

#     def define_deps(self):
#         self.register_dependencies([
#             (self.layer1, self.bnorm1, DepType.SAME),
#             (self.bnorm1, self.layer2, DepType.INCOMING),
#             (self.layer2, self.bnorm2, DepType.SAME),
#             (self.bnorm2, self.fc1, DepType.INCOMING),
#             (self.fc1, self.fc2, DepType.INCOMING),
#             (self.fc2, self.fc3, DepType.INCOMING),
#         ])

#         self.flatten_conv_id = self.bnorm2.get_module_id()

#     def forward(self, x, intermediary: Dict[int, th.Tensor] | None = None):
#         self.maybe_update_age(x)
        
#         x = self.layer1(x, intermediary=intermediary)
#         x = self.bnorm1(x)
#         x = F.relu(x)
#         x = self.mpool1(x)

#         x = self.layer2(x, intermediary=intermediary)
#         x = self.bnorm2(x)
#         x = F.relu(x)
#         x = self.mpool2(x)

#         x = x.view(x.size(0), -1)
#         x = self.fc1(x, intermediary=intermediary)
#         x = F.relu(x)
#         # x = self.bnorm3(x)
#         # x = self.drop1(x)
#         x = self.fc2(x, intermediary=intermediary)
#         x = F.relu(x)
#         # x = self.bnorm4(x)
#         output = self.fc3(x, skip_register=True, intermediary=None)

#         one_hot = F.one_hot(
#             output.argmax(dim=1), num_classes=self.fc3.out_features)

#         if hasattr(x, 'in_id_batch') and hasattr(x, 'label_batch'):
#             add_tracked_attrs_to_input_tensor(
#                 one_hot, in_id_batch=input.in_id_batch,
#                 label_batch=input.label_batch)
#         self.fc3.register(one_hot)
#         output = self.softmax(output)
#         return output


import numpy as np, pandas as pd
import matplotlib.pyplot as plt


test = pd.read_csv("/home/rotaru/Desktop/GRAYBOX/repos/weightslab_ui/data/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/home/rotaru/Desktop/GRAYBOX/repos/weightslab_ui/data/equity-post-HCT-survival-predictions/train.csv")


train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += 2*len(train)
train.y = train.y / train.y.max()
train.y = np.log( train.y )
train.y -= train.y.mean()
train.y *= -1.0


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
        CATS.append(c)
    elif not "age" in c:
        train[c] = train[c].astype("str")
        test[c] = test[c].astype("str")
        CATS.append(c)
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


CAT_SIZE = []
CAT_EMB = []
NUMS = []

combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

print("We LABEL ENCODE the CATEGORICAL FEATURES: ")

for c in FEATURES:
    if c in CATS:
        # LABEL ENCODE
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        #combined[c] = combined[c].astype("category")

        n = combined[c].nunique()
        mn = combined[c].min()
        mx = combined[c].max()
        print(f'{c} has ({n}) unique values')

        CAT_SIZE.append(mx+1) 
        CAT_EMB.append( int(np.ceil( np.sqrt(mx+1))) ) 
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")

        m = combined[c].mean()
        s = combined[c].std()
        combined[c] = (combined[c]-m)/s
        combined[c] = combined[c].fillna(0)

        NUMS.append(c)

train_df = combined.iloc[:len(train)].copy()
test_df = combined.iloc[len(train):].reset_index(drop=True).copy()



class TabularModel(NetworkWithOps):
    def __init__(self, cat_sizes, cat_embs, num_features):
        super(TabularModel, self).__init__()

        # Categorical Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_size, emb_dim) for cat_size, emb_dim in zip(cat_sizes, cat_embs)
        ])

        input_dim = sum(cat_embs) + len(num_features)
        self.fc1 = LinearWithNeuronOps(input_dim, 256)
        self.fc2 = LinearWithNeuronOps(256, 256)
        self.fc3 = LinearWithNeuronOps(256, 1)

    def define_deps(self):
        self.register_dependencies([
            (self.fc1, self.fc2, DepType.INCOMING),
            (self.fc2, self.fc3, DepType.INCOMING),
        ])

    def children(self):
        return [self.fc1, self.fc2, self.fc3]

    def forward(self, data):
        # import pdb; pdb.set_trace()
        x_features = data[:len(self.embeddings)]
        x_nums = data[len(self.embeddings):]
        embs = []
        for i, emb in enumerate(self.embeddings):
            e = emb(x_features[i].long())
            e = e.view(e.size(0), -1)  # Flatten embedding
            embs.append(e)

        x_nums = [x_num.view(x_num.size(0), -1).double() for x_num in x_nums]
        embs.extend(x_nums)
        x = torch.cat(embs, dim=1)
        x = x.float()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Linear output for regression
        return x


class HCT_Kaggle_Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        category_data = self.df.loc[idx, CATS].values
        numerical_data = train.loc[idx, NUMS].values
        data = list(category_data) + list(numerical_data)

        target1 = train.loc[idx, "y"]
        # target2 = train.loc[idx, "efs"]
        # print("dataframe ", idx, data, target1)
        return  data, target1


train_set = HCT_Kaggle_Dataset(train_df)
test_set = HCT_Kaggle_Dataset(test_df)


print("train_set:", len(train_set))
print("test_set:", len(test_set))


# device = th.device("cuda:0")
device = th.device("cpu")


def get_exp():
    print("GET EXP")
    print(CAT_SIZE, CAT_EMB, NUMS)
    model = TabularModel(CAT_SIZE, CAT_EMB, NUMS)
    model.define_deps()
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=train_set,
        eval_dataset=test_set,
        device=device, learning_rate=1e-3, batch_size=100,
        name="v0",
        root_log_dir='hct-dev',
        logger=Dash("hct-dev"),
        skip_loading=False,
        criterion=nn.MSELoss(reduction="none"))

    def stateful_difference_monitor_callback():
        exp.display_stats()

    exp.register_train_loop_callback(stateful_difference_monitor_callback)

    return exp

exp = get_exp()
exp.train_n_steps_with_eval_full(1000)