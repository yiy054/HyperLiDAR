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

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

parser = argparse.ArgumentParser()
parser.add_argument('-stop', '--layers', type=int, help='how many layers deep', default=0)
args = parser.parse_args()

wandb.login()

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

# Load pretrained model
ckpt = torch.load('/root/main/ScaLR/saved_models/ckpt_last_scalr.pth', map_location=device_string) # cuda:0
ckpt = ckpt["net"]

new_ckpt = {}
for k in ckpt.keys():
    if k.startswith("module"):
        if k.startswith("module.classif.0"):
            continue
        elif k.startswith("module.classif.1"):
            new_ckpt["classif" + k[len("module.classif.1") :]] = ckpt[k]
        else:
            new_ckpt[k[len("module.") :]] = ckpt[k]
    else:
        new_ckpt[k] = ckpt[k]

model.load_state_dict(new_ckpt)

if device_string != 'cpu':
    torch.cuda.set_device(device_string) # cuda:0
    model = model.cuda(device_string) # cuda:0

model.eval()

kwargs = {
        "rootdir": '/root/main/dataset/nuscenes',
        "input_feat": ["xyz", "intensity", "radius"],
        "voxel_size": 0.1,
        "num_neighbors": 16,
        "dim_proj": [2, 1, 0],
        "grids_shape": [[256, 256], [256, 32], [256, 32]],
        "fov_xyz": [[-64, -64, -8], [64, 64, 8]], # Check here
    }

# Get datatset
DATASET = LIST_DATASETS.get("nuscenes")

# Train dataset
train_dataset = DATASET(
    phase="train",
    **kwargs,
)

test_dataset = DATASET(
    phase="val",
    **kwargs,
)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=Collate(),
    )

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=Collate(),
    )

DIMENSIONS = 10000
FEAT_SIZE = 768
NUM_LEVELS = 1000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones


class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


encode = Encoder(DIMENSIONS, FEAT_SIZE, NUM_LEVELS)
encode = encode.to(device)

num_classes = 16
model_hd = Centroid(DIMENSIONS, num_classes)
model_hd = model_hd.to(device)

stop = args.layers

run = wandb.init(
    # Set the project where this run will be logged
    project="scalr_hd",
    # Track hyperparameters and run metadata
    config={
        "encoding": "Random * Level 1000",
        "hd_dim": 10000,
        "training_samples": len(train_loader),
    },
    id=f"hd_param_stop_layer_{stop}",
    key=["9487c04b8eff0c16cac4e785f2b57c3a475767d3"],
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

    with torch.no_grad():
        out = model(feat, cell_ind, occupied_cell, neighbors_emb, stop)
        embed, tokens = out[0][0], out[1][0]
        embed = embed.transpose(0, 1)
        tokens = tokens.transpose(0, 1)

        labels_v = [[] for i in range(embed.shape[0])]
        for i, vox in enumerate(batch["upsample"][0]):
            labels_v[vox].append(labels[i])
        labels_v_single = []
        for labels_ in labels_v:
            lab_tens = torch.tensor(labels_)
            most_common_value = torch.bincount(lab_tens).argmax()
            labels_v_single.append(most_common_value)
    
    return tokens, labels_v_single

def val(stop):
    #accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    miou = MulticlassJaccardIndex(num_classes=16, average=None)

    model_hd.normalize()
    
    output_array = []
    labels_array = []

    for it, batch in enumerate(test_loader):
        if it < 10:
            # Network inputs
            
                tokens, labels_v_single = forward_model(it, batch, stop)

                #HD Testing
                for samples, l in tqdm(zip(tokens,labels_v_single), desc="Testing"):
                    if l != 255: # Make sure its not noise
                        
                        samples = samples.to(device)
                        samples_hv = encode(samples).reshape((1, DIMENSIONS))
                        outputs = model_hd(samples_hv, dot=True)
                        outputs = outputs.argmax().data#, device=device_string).reshape((1))
                        #l = torch.tensor([l])
                        #accuracy.update(outputs.cpu(), l)
                        output_array.append(outputs.cpu())
                        labels_array.append(l)

        else:
            break
    
    l = torch.tensor(labels_array)
    out = torch.tensor(output_array)

    accuracy = miou(out, l)
    mean = torch.mean(accuracy)

    print(f"Mean accuracy of {mean}")
    log_data = {f"class_{i}_IoU": c for i, c in enumerate(accuracy)}
    log_data["meanIoU"] = mean
    wandb.log(log_data)

for it, batch in enumerate(train_loader):
    
    # Network inputs
    
    tokens, labels_v_single = forward_model(it, batch, stop)

    #HD Training
    for samples, labels in tqdm(zip(tokens,labels_v_single), desc="Training"):
        if labels != 255:
            samples = samples.to(device)
            labels = labels.to(device)
            samples_hv = encode(samples).reshape((1, DIMENSIONS))
            model_hd.add(samples_hv, labels)


    # Voxels to points
    #token_upsample = []
    #temp = None
    #for id_b, closest_point in enumerate(batch["upsample"]):
    #    temp = tokens[id_b, :, closest_point]
    #    token_upsample.append(temp.T)
    #token_2 = torch.cat(token_upsample, dim=0)
    
    val(stop)