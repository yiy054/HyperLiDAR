import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

INPUT_DIM = 768
HD_DIM = 10000
num_classes = 19

class Encoder(nn.Module):
    def __init__(self, hd_dim, size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.projection = embeddings.Projection(size, hd_dim)
        #self.position = embeddings.Random(size * size, out_features)
        #self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        #x = self.flatten(x)
        #sample_hv = torchhd.bind(self.position.weight, self.value(x))
        #sample_hv = torchhd.multiset(sample_hv)
        sample_hv = self.projection(x)
        return torchhd.hard_quantize(sample_hv)

encode = Encoder(HD_DIM, INPUT_DIM)
encode = encode.to(device)

model = Centroid(HD_DIM, num_classes)
model = model.to(device)

# Metric
miou = MulticlassJaccardIndex(num_classes=19, average=None).to(device)

#arrays = np.load('SoA_results.npy')
features = torch.load('/root/main/ScaLR/debug/semantic_kitti/feat_train_semkitti.pt')
labels = torch.load('/root/main/ScaLR/debug/semantic_kitti/labels_train_semkitti.pt')
num_voxels = torch.load('/root/main/ScaLR/debug/semantic_kitti/voxels_train_semkitti.pt')
#print(arrays)

def normalize(samples, min_val=None, max_val=None):
    # normalize # 0 -> 768 # 1 -> 16487

    ##### MIN MAX #####

    #if min_val == None:
    #    min_val = torch.min(samples, axis=0).values
    #if max_val == None:
    #    max_val = torch.max(samples, axis=0).values
    #samples = (samples - min_val) / (max_val - min_val)
    #for s in range(len(samples)):
    #    samples[s] = (samples[s] - min_val[s]) / (max_val[s] - min_val[s])
    
    ##### SOFTMAX #####

    #m = nn.Softmax(dim=0)
    #first_sample = m(first_sample)
    #print(first_sample)

    ### Z-Score ####

    # Compute mean and std for each feature (dim=0 for columns)
    mean = torch.mean(samples, dim=0)
    std = torch.std(samples, dim=0)

    samples = (samples - mean) / (std + 1e-8)

    return samples

# Minimizing over all the samples
'''
min_val = torch.zeros((768))
max_val = torch.zeros((768))

for i in range(num_samples):
    first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(device)
    min_here = torch.min(first_sample, axis=0).values
    max_here = torch.max(first_sample, axis=0).values
    for f in range(len(min_here)):
        if min_here[f] < min_val[f]:
            min_val[f] = min_here[f]
        if max_here[f] > max_here[f]:
            max_val[f] = max_here[f]
'''

"""CLASS_NAME = [
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ]""" # Nuscenes

CLASS_NAME = [
        "car",  # 0
        "bicycle",  # 1
        "motorcycle",  # 2
        "truck",  # 3
        "other-vehicle",  # 4
        "person",  # 5
        "bicyclist",  # 6
        "motorcyclist",  # 7
        "road",  # 8
        "parking",  # 9
        "sidewalk",  # 10
        "other-ground",  # 11
        "building",  # 12
        "fence",  # 13
        "vegetation",  # 14
        "trunk",  # 15
        "terrain",  # 16
        "pole",  # 17
        "traffic-sign",  # 18
    ] # Semantic - kitti



#####################################################
# Training and Inference
#####################################################

def train(features, labels, num_voxels, device):
    num_samples = len(features)
    for i in range(num_samples):
        # compute the accuracy of the one sample
        first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(device)
        first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int64).to(device)

        #pred_ts = torch.Tensor(np.argmax(first_sample, axis=1)).to(device)
        #label_ts = torch.Tensor(first_label).to(torch.int32).to(device)

        first_sample = normalize(first_sample) # min_val, max_val

        # HD training
        samples_hv = encode(first_sample)
        model.add_online(samples_hv, first_label, lr=0.00001)

        # HD prediction
        #pred_hd = model(samples_hv, dot=True).argmax(1).data

        #print('pred_ts', pred_ts)
        #print('pred_hd', pred_hd)
        #print('label', first_label)
        #accuracy = miou(pred_hd, first_label)
        #avg_acc = torch.mean(accuracy)
        #print(f'accuracy of sample {i}: {accuracy}')
        #print(f'avg acc of sample {i}: {avg_acc}')


#####################################################
# Pure inference
#####################################################
for i in range(101):
    # compute the accuracy of the one sample
    first_sample = features[i][:,:int(num_voxels[i])].to(device)
    first_label = labels[i][:int(num_voxels[i])].to(torch.int32).to(device)
    first_sample = torch.transpose(first_sample, 0, 1)

    first_sample = normalize(first_sample) # min_val, max_val

    #pred_ts = torch.Tensor(np.argmax(first_sample, axis=1)).to(device)
    #label_ts = torch.Tensor(first_label).to(torch.int32).to(device)

    # HD training
    samples_hv = encode(first_sample)
    #model.add(samples_hv, first_label)

    # HD prediction
    pred_hd = model(samples_hv, dot=True).argmax(1).data

    #print('pred_ts', pred_ts)
    print('pred_hd', pred_hd)
    print('label', first_label)
    accuracy = miou(pred_hd, first_label)
    cm = confusion_matrix(pred_hd, first_label, labels=torch.Tensor(range(0,15)))
    print("Confusion matrix \n")
    print(cm)
    avg_acc = torch.mean(accuracy)
    print(f'accuracy of sample {i}: {accuracy}')
    print(f'avg acc of sample {i}: {avg_acc}')
    print("Similarity between HVs")
    #print(torchhd.cosine_similarity(model.weight, model.weight))