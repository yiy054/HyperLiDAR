from models.waffleiron.segmenter import Segmenter
import torch
from datasets import LIST_DATASETS, Collate
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import wandb
from torchmetrics.classification import MulticlassJaccardIndex

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm
from collections import OrderedDict
import warnings
import copy
import random
import numpy as np
import os
from sklearn.metrics import confusion_matrix

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings


parser = argparse.ArgumentParser()
parser.add_argument('-stop', '--layers', type=int, help='how many layers deep', default=48)
parser.add_argument('-soa', '--soa', action="store_true", default=False, help='Plot SOA')
parser.add_argument('-number_samples', '--number_samples', type=int, help='how many scans to train', default=10000)
parser.add_argument(
        "--seed", default=None, type=int, help="Seed for initializing training"
    )
parser.add_argument('-val', '--val', action="store_true", default=False, help='Train with validation for each scan')
args = parser.parse_args()

# Set seed
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

wandb.login(key="9487c04b8eff0c16cac4e785f2b57c3a475767d3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
device_string = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Segmenter(
    input_channels=5,
    feat_channels=768,
    depth=48,
    grid_shape=[[256, 256], [256, 32], [256, 32]],
    nb_class=16, # class for prediction
    #drop_path_prob=config["waffleiron"]["drop_path"],
    layer_norm=True,
)

classif = torch.nn.Conv1d(
    768, 16, 1
)
torch.nn.init.constant_(classif.bias, 0)
torch.nn.init.constant_(classif.weight, 0)
model.classif = torch.nn.Sequential(
    torch.nn.BatchNorm1d(768),
    classif,
)

for p in model.parameters():
    p.requires_grad = False
for p in model.classif.parameters():
    p.requires_grad = True

def get_optimizer(parameters):
    return torch.optim.AdamW(
        parameters,
        lr=0.001,
        weight_decay=0.003,
    )

optim = get_optimizer(model.parameters())

# Load pretrained model
path_to_ckpt = '/root/main/ScaLR/saved_models/ckpt_last_scalr.pth'
checkpoint = torch.load(path_to_ckpt,
    map_location=device_string)
state_dict = checkpoint["net"]  # Adjust key as needed
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # Remove "module." prefix
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)

print(
    f"Checkpoint loaded on {device_string}: {path_to_ckpt}"
)

if device_string != 'cpu':
    torch.cuda.set_device(device_string) # cuda:0
    model = model.cuda(device_string) # cuda:0

model.eval()

kwargs = {
        "rootdir": '/root/main/dataset/nuscenes',
        "input_feat": ["intensity", "xyz", "radius"],
        "voxel_size": 0.1,
        "num_neighbors": 16,
        "dim_proj": [2, 1, 0],
        "grids_shape": [[256, 256], [256, 32], [256, 32]],
        "fov_xyz": [[-64, -64, -8], [64, 64, 8]], # Check here
    }

# Get datatset
DATASET = LIST_DATASETS.get("nuscenes")

# Train dataset
dataset = DATASET(
    phase="train",
    **kwargs,
)
#print(dataset.voxel_size)
dataset_train = copy.deepcopy(dataset)
dataset_val = copy.deepcopy(dataset)
dataset_train.init_training()
dataset_val.init_val()

train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=Collate(),
    )

val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=Collate(),
    )

DIMENSIONS = 10000
FEAT_SIZE = 16
NUM_LEVELS = 8000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels, device=torch.device("cuda:0")):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        #self.position = embeddings.Random(size, out_features)
        #self.value = embeddings.Level(levels, out_features)
        self.rp = torchhd.embeddings.Projection(size, out_features, device=device)
        self.rp = self.rp.to(device)
        self.device = device

    def forward(self, x):
        # Find the min and max values
        #min_val = torch.min(x)
        #max_val = torch.max(x)

        # Normalize the tensor
        #norm_x = (x - min_val) / (max_val - min_val)
        #print(x.shape)
        #print(x.type())
        projected = self.rp(x.type(torch.FloatTensor).to(self.device))
        #sample_hv = torchhd.bind(self.position.weight, self.value(norm_x))
        #sample_hv = torchhd.multiset(sample_hv)
        #hv_all = torch.sum(hv_all, dim=0).sign()
        return torchhd.hard_quantize(projected)


encode = Encoder(DIMENSIONS, FEAT_SIZE, NUM_LEVELS, device=device)
encode = encode.to(device)

num_classes = 16
model_hd = Centroid(DIMENSIONS, num_classes)
model_hd = model_hd.to(device)

if not args.soa:
    stop = args.layers
    name = f"hd_param_layer_{stop}"
else:
    stop = 48
    name = f"ScaLR_SoA"

run = wandb.init(
    # Set the project where this run will be logged
    project="scalr_hd",
    # Track hyperparameters and run metadata
    config={
        "encoding": "Random * Level 1000",
        "hd_dim": 10000,
        "training_samples": len(train_loader),
    },
    id=name,
)

def forward_model(it, batch, stop):
    feat = batch["feat"]
    labels = batch["labels_orig"]
    cell_ind = batch["cell_ind"]
    occupied_cell = batch["occupied_cells"]
    neighbors_emb = batch["neighbors_emb"]
    if device_string != 'cpu':
        feat = feat.cuda(0, non_blocking=True)
        labels = labels.cuda(0, non_blocking=True)
        batch["upsample"] = [
            up.cuda(0, non_blocking=True) for up in batch["upsample"]
        ]
        cell_ind = cell_ind.cuda(0, non_blocking=True)
        occupied_cell = occupied_cell.cuda(0, non_blocking=True)
        neighbors_emb = neighbors_emb.cuda(0, non_blocking=True)
    net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

    with torch.autocast("cuda", enabled=True):
        # Logits
        with torch.no_grad():
            out = model(*net_inputs, stop)
            encode, tokens, out = out[0], out[1], out[2]
            pred_label = out.max(1)[1]

    # Confusion matrix
    with torch.no_grad():
        nb_class = out.shape[1]
        where = labels != 255
    
    return tokens[0,:,where], labels[where], pred_label[0, where]

def val(stop):
    #accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    
    output_array = []
    labels_array = []

    for it, batch in tqdm(enumerate(val_loader), desc="Testing"):
        # Network inputs
        
        tokens, labels, full = forward_model(it, batch, stop)
        tokens = torch.transpose(tokens, 0,1)

        #HD Testing
        if not args.soa:
            for samples, l in zip(tokens,labels):
                
                samples = samples.to(device)
                samples_hv = encode(samples).reshape((1, DIMENSIONS))
                outputs = model_hd(samples_hv, dot=True)
                outputs = outputs.argmax().data#, device=device_string).reshape((1))
                #l = torch.tensor([l])
                #accuracy.update(outputs.cpu(), l)
                output_array.append(outputs.cpu())
                labels_array.append(l)
    
    if not args.soa:
        l = torch.tensor(labels_array)
        out = torch.tensor(output_array)
    else:
        l = labels.cpu()
        out = full.cpu()

    accuracy = miou(out, l)
    mean = torch.mean(accuracy)

    print(f"Validation Mean accuracy of {mean}")
    log_data = {f"Validation class_{i}_IoU": c for i, c in enumerate(accuracy)}
    log_data["Validation meanIoU"] = mean
    wandb.log(log_data)

#num_samples_per_class = {}

#def intelligent_sampling(tokens, labels_v_single):

#    return None

output_array_t = []
labels_array_t = []

miou = MulticlassJaccardIndex(num_classes=16, average=None)

# Train

for it, batch in tqdm(enumerate(train_loader), desc="Training"):
    
    # Network inputs
    
    tokens, labels, soa_result = forward_model(it, batch, stop)
    #training_ids = intelligent_sampling(tokens, labels_v_single)
    #tokens, labels_v_single = tokens[training_ids], labels_v_single[training_ids]
    tokens = torch.transpose(tokens, 0,1)
    
    # If you are returning the full arrey of (16, voxels) then transpose
    # soa_result = torch.transpose(soa_result, 0,1)

    #HD Training
    if not args.soa:
        for samples, lab in zip(tokens,labels):
            samples = samples.to(device)
            lab = lab.to(device).reshape((1,))
            samples_hv = encode(samples).reshape((1, DIMENSIONS))
            model_hd.add_online(samples_hv, lab, lr=0.01) # Lr change

        for samples, lab in zip(tokens,labels):
            samples = samples.to(device)
            lab = lab.to(device).reshape((1,))
            samples_hv = encode(samples).reshape((1, DIMENSIONS))
            outputs = model_hd(samples_hv, dot=True)
            outputs = outputs.argmax().data#, device=device_string).reshape((1))
            output_array_t.append(outputs.cpu())
            labels_array_t.append(lab)

        #x = input()
    if args.val:
        if it % 10 == 0: # Test every 10 samples
            val(stop)

    if not args.soa:
        l = torch.tensor(labels_array_t)
        out = torch.tensor(output_array_t)
    else:
        l = labels.cpu()
        out = soa_result.cpu()

    accuracy = miou(out, l)
    mean = torch.mean(accuracy)
    # Generate confusion matrix
    cm = confusion_matrix(out, l)

    # PRINT RESULTS 

    # Convert confusion matrix to string for saving
    cm_str = "\n".join(["\t".join(map(str, row)) for row in cm])
    # Save to a text file
    file_path = f"{name}_results.txt"
    with open(file_path, "a") as f:
        f.write(f"Confusion Matrix fo sample {it}:\n")
        f.write(cm_str)

    print(f"Training mean accuracy of {mean}")
    log_data = {f"Training class_{i}_IoU": c for i, c in enumerate(accuracy)}
    log_data["Training meanIoU"] = mean
    wandb.log(log_data)

    if it == args.number_samples:
        break