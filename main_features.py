import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

INPUT_DIM = 768
HD_DIM = 10000
num_classes = 16

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
miou = MulticlassJaccardIndex(num_classes=16, average=None).to(device)

#arrays = np.load('SoA_results.npy')
features = np.load('/home/outputs/SoA_features.npy')
labels = np.load('/home/outputs/SoA_labels.npy')
num_voxels = np.load('/home/outputs/num_voxels.npy')
#print(arrays)

num_samples = len(features)

def normalize(samples):
    # normalize
    min_val = torch.min(samples, axis=0).values
    print(min_val.shape)
    max_val = torch.max(samples, axis=0).values
    #samples = (samples - min_val) / (max_val - min_val)
    for s in range(len(samples)):
        samples[s] = (samples[s] - min_val[s]) / (max_val[s] - min_val[s])
    #m = nn.Softmax(dim=0)
    #first_sample = m(first_sample)
    #print(first_sample)
    return samples


for i in range(num_samples):
    # compute the accuracy of the one sample
    first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(device)
    first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int32).to(device)

    #pred_ts = torch.Tensor(np.argmax(first_sample, axis=1)).to(device)
    #label_ts = torch.Tensor(first_label).to(torch.int32).to(device)

    first_sample = normalize(first_sample)

    # HD training
    samples_hv = encode(first_sample)
    model.add(samples_hv, first_label)

    # HD prediction
    pred_hd = model(samples_hv, dot=True).argmax(1).data

    #print('pred_ts', pred_ts)
    print('pred_hd', pred_hd)
    print('label', first_label)
    accuracy = miou(pred_hd, first_label)
    avg_acc = torch.mean(accuracy)
    print(f'accuracy of sample {i}: {accuracy}')
    print(f'avg acc of sample {i}: {avg_acc}')


#####################################################
# Pure inference
#####################################################
for i in range(num_samples):
    # compute the accuracy of the one sample
    first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(device)
    first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int32).to(device)

    first_sample = normalize(first_sample)

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
    avg_acc = torch.mean(accuracy)
    print(f'accuracy of sample {i}: {accuracy}')
    print(f'avg acc of sample {i}: {avg_acc}')





"""
accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

with torch.no_grad():
    model.normalize()

    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=True)
        accuracy.update(outputs.cpu(), labels)
"""