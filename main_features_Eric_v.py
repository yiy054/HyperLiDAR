import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.preprocessing import Normalizer
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, hd_dim, size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.projection = embeddings.Projection(size, hd_dim)

    def forward(self, x):
        sample_hv = self.projection(x)
        return torchhd.hard_quantize(sample_hv)


class HD_Model:
    def __init__(self, in_dim, out_dim, num_classes, device):

        encode = Encoder(out_dim, in_dim)
        self.encode = encode.to(device)

        model = Centroid(out_dim, num_classes)
        self.model = model.to(device)
        self.device = device

    def normalize(self, samples):

        """ Normalize with Z-score"""

        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)

        samples = (samples - mean) / (std + 1e-8)

        return samples

    def train(self, features, labels, num_voxels):

        """ Initial training pass """

        assert len(features) == len(labels)

        print("\nTrain First\n")

        for i in range(len(features)):
            first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(self.device)
            first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int64).to(self.device)

            first_sample = self.normalize(first_sample) # Z1 score seems to work

            #for vox in range(len(first_sample)):
                
            #    label = first_label[vox]
            #    if vox % 5000 == 0:
            #        print(f"Sample {i}: Voxel {vox}")
                
            # HD training
            samples_hv = self.encode(first_sample)
            #samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
            self.model.add(samples_hv, first_label)

    def retrain(self, features, labels, num_voxels):
        for e in tqdm(range(10), desc="Epoch"):
            count = 0

            for i in range(len(features)):
                first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(self.device)
                first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int64).to(self.device)

                first_sample = self.normalize(first_sample) # Z1 score seems to work

                for vox in range(len(first_sample)):
                    # HD training
                    samples_hv = self.encode(first_sample[vox])
                    pred_hd = model(samples_hv, dot=True).argmax(1).data
                    # When we need the similarites
                    #similarities = cosine_similarity(samples_hv, self.model.weight)[0]
                    #pred = np.argmax(similarities)
                    label = first_label[vox]

                    if pred_hd != label:
                        self.weight.index_add_(0, label, samples_hv, alpha=1.0)
                        self.weight.index_add_(0, pred_hd, samples_hv, alpha=-1.0)
                        count += 1
            print("Misclassified: ", count)

    def test(self, features, labels, num_voxels):
        assert len(features) == len(labels)
        
        # Metric
        miou = MulticlassJaccardIndex(num_classes=16, average=None).to(self.device)
        
        for i in range(features):
            first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(self.device)
            first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int64).to(self.device)

            first_sample = self.normalize(first_sample) # Z1 score seems to work

            # HD inference
            samples_hv = self.encode(first_sample)
            pred_hd = self.model(samples_hv, dot=True).argmax(1).data

            print("================================")

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

            print("================================")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    INPUT_DIM = 768
    HD_DIM = 10000
    num_classes = 16

    
    # Loading the data
    features = np.load('/home/outputs/SoA_features.npy')
    labels = np.load('/home/outputs/SoA_labels.npy')
    num_voxels = np.load('/home/outputs/num_voxels.npy')

    model = HD_Model(INPUT_DIM, HD_DIM, num_classes, device)

    model.train(features, labels, num_voxels)
    model.test(features, labels, num_voxels)
    model.retrain(features, labels, num_voxels)
    model.test(features, labels, num_voxels)
