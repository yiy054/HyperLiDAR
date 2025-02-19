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
import matplotlib.pyplot as plt
import seaborn as sns
from tsnecuda import TSNE

from tqdm import tqdm
import wandb

class Encoder(nn.Module):
    def __init__(self, hd_dim, size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.projection = embeddings.Projection(size, hd_dim)
        self.projection.weight = nn.Parameter(torchhd.normalize(self.projection.weight), requires_grad=False) # Binary

    def forward(self, x):
        sample_hv = self.projection(x)
        return torchhd.hard_quantize(sample_hv).to(torch.int32)


class HD_Model:
    def __init__(self, in_dim, out_dim, num_classes, device):

        encode = Encoder(out_dim, in_dim)
        self.encode = encode.to(device)

        model = Centroid(out_dim, num_classes, dtype=torch.int32)
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.hd_dim = out_dim

    def normalize(self, samples):

        """ Normalize with Z-score"""

        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)

        #print("Mean in range: ", min(mean), " ", max(mean))
        #print("Std in range: ", min(std), " ", max(std))

        samples = (samples - mean) / (std + 1e-8)

        """ Min - max"""

        #min_val = samples.min()
        #max_val = samples.max()

        # Perform Min-Max normalization
        #normalized_tensor = (samples - min_val) / (max_val - min_val)
        #samples = normalized_tensor * (max_range - min_range) + min_range

        return samples

    def train(self, features, labels, num_voxels):

        """ Initial training pass """

        assert len(features) == len(labels)

        print("\nTrain First\n")
        batch = 15000

        for i in tqdm(range(len(features)), desc="1st Training:"):
            for b in range(0,int(num_voxels[i]), batch):
                end = min(b + batch, int(num_voxels[i]))  # Ensure we don't exceed num_voxels[i]
                first_sample = torch.Tensor(features[i][b:end]).to(self.device)
                first_label = torch.Tensor(labels[i][b:end]).to(torch.int32).to(self.device)

                first_sample = self.normalize(first_sample) # Z1 score seems to work

                #for vox in range(len(first_sample)):
                    
                #    label = first_label[vox]
                #    if vox % 5000 == 0:
                #        print(f"Sample {i}: Voxel {vox}")
                    
                # HD training

                samples_hv = self.encode(first_sample).to(torch.int32)
                
                #### Original ####
                #temp = torch.zeros(self.num_classes, self.hd_dim, dtype=torch.int32).to(self.device)
                #temp.index_add_(0, first_label, samples_hv)
                #print("Min: ", torch.min(temp), "\nMax: ", torch.max(temp))
                #temp = temp
                # Add the 16 bit integer
                #self.model.weight = nn.Parameter(self.model.weight + temp, requires_grad=False) # Addition
                self.model.add(samples_hv, first_label)

        # Normalizing works way better :)
        #self.model.normalize() # Min Max
        self.model.weight = nn.Parameter(torchhd.normalize(self.model.weight), requires_grad=False) # Binary

    def retrain(self, features, labels, num_voxels):
        
        """ Retrain with misclassified samples (also substract)"""

        batch = 15000
        
        for e in tqdm(range(10), desc="Epoch"):
            count = 0

            for i in range(len(features)):
                for b in range(0,int(num_voxels[i]), batch):
                    end = min(b + batch, int(num_voxels[i]))  # Ensure we don't exceed num_voxels[i]
                    first_sample = torch.Tensor(features[i][b:end]).to(self.device)
                    first_label = torch.Tensor(labels[i][b:end]).to(torch.int32).to(self.device)
                    first_sample = self.normalize(first_sample) # Z1 score seems to work

                    #for vox in range(len(first_sample)):
                    samples_hv = self.encode(first_sample)
                    sim = self.model(samples_hv, dot=True)
                    pred_hd = sim.argmax(1).data

                    is_wrong = first_label != pred_hd

                    # cancel update if all predictions were correct
                    if is_wrong.sum().item() == 0:
                        continue

                    # only update wrongly predicted inputs
                    samples_hv = samples_hv[is_wrong]
                    first_label = first_label[is_wrong]
                    pred_hd = pred_hd[is_wrong]

                    #count = first_label.shape[0]
                            
                    #print(f"Misclassified for {i}: ", count)

                    ## Original ###
                    self.model.weight.index_add_(0, first_label, samples_hv)
                    self.model.weight.index_add_(0, pred_hd, samples_hv, alpha=-1)

            # If you want to test for each sample
            #print(self.model.weight) # Int it is I think...
            #self.model.weight = nn.Parameter(torch.clamp(self.model.weight, min=-128, max=127).to(torch.int8), requires_grad=False)
            #print("Min model: ", torch.min(self.model.weight), "\nMax model: ", torch.max(self.model.weight))
            #print(self.model.weight)
            self.test_hd(features, labels, num_voxels, e+1)

    def test_hd(self, features, labels, num_voxels, epoch=0):

        """ Testing over all the samples in all the scans given """

        assert len(features) == len(labels)

        batch = 15000
        
        # Metric
        miou = MulticlassJaccardIndex(num_classes=16, average=None).to(self.device)
        final_shape = int(np.sum(num_voxels))
        final_labels = torch.empty((final_shape), device=self.device)
        final_pred = torch.empty((final_shape), device=self.device)
        
        start_idx = 0
        for i in tqdm(range(len(features)), desc="Testing"):
            for b in range(0,int(num_voxels[i]), batch):
                end = min(b + batch, int(num_voxels[i]))  # Ensure we don't exceed num_voxels[i]
                first_sample = torch.Tensor(features[i][b:end]).to(self.device)
                first_label = torch.Tensor(labels[i][b:end]).to(torch.int64)
                final_labels[start_idx:start_idx+end-b] = first_label

                first_sample = self.normalize(first_sample) # Z1 score seems to work

                # HD inference
                samples_hv = self.encode(first_sample)
                pred_hd = self.model(samples_hv, dot=True).argmax(1).data
                final_pred[start_idx:start_idx+end-b] = pred_hd

                start_idx += end-b

        print("================================")

        #print('pred_ts', pred_ts)
        print('pred_hd', final_pred, "\tShape: ", final_pred.shape)
        print('label', final_labels, "\tShape: ", final_labels.shape)
        accuracy = miou(final_pred, final_labels)
        avg_acc = torch.mean(accuracy)
        print(f'accuracy: {accuracy}')
        print(f'avg acc: {avg_acc}')

        #log_data = {f"Training class_{i}_IoU": c for i, c in enumerate(accuracy)}
        #log_data["Retraining epoch"] = avg_acc
        #wandb.log(log_data)

        # Compute the confusion matrix
        cm = confusion_matrix(final_labels.cpu().numpy(), final_pred.cpu().numpy(), labels=torch.arange(16).numpy())

        # Plot the confusion matrix
        plt.figure(figsize=(16, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(16), yticklabels=range(16))
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for Epoch {epoch}")

        # Save the figure
        plt.savefig(f"nuscenes_confusion_matrix_{epoch}.png", dpi=300, bbox_inches="tight")

        print("================================")

def test_soa(results, labels, num_voxels, device):
    assert len(results) == len(labels)
        
    # Metric
    miou = MulticlassJaccardIndex(num_classes=16, average=None).to(device)
    final_shape = int(np.sum(num_voxels))
    final_labels = torch.empty((final_shape), device=device)
    final_pred = torch.empty((final_shape), device=device)
    
    start_idx = 0
    for i in tqdm(range(len(results)), desc="Testing SoA"):
        shape_sample = int(num_voxels[i])
        first_sample = torch.Tensor(results[i][:shape_sample]).to(device)
        first_label = torch.Tensor(labels[i][:shape_sample]).to(torch.int64)
        final_labels[start_idx:start_idx+shape_sample] = first_label

        pred = first_sample.max(1)[1]
        final_pred[start_idx:start_idx+shape_sample] = pred

        start_idx += shape_sample

    print("================================")

    print('pred', final_pred, "\tShape: ", final_pred.shape)
    print('label', final_labels, "\tShape: ", final_labels.shape)
    accuracy = miou(final_pred, final_labels)
    avg_acc = torch.mean(accuracy)
    print(f'accuracy: {accuracy}')
    print(f'avg acc: {avg_acc}')

    #cm = confusion_matrix(pred_hd, first_label, labels=torch.Tensor(range(0,15)))
    #print("Confusion matrix \n")
    #print(cm)

    print("================================")


if __name__ == "__main__":

    #wandb.login(key="9487c04b8eff0c16cac4e785f2b57c3a475767d3")

    """run = wandb.init(
        # Set the project where this run will be logged
        project="scalr_hd",
        # Track hyperparameters and run metadata
        config={
            "encoding": "Random Projection",
            "hd_dim": 10000,
            "training_samples": 10,
        },
        id="lr_imbalance_hd_simple1",
    )"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    INPUT_DIM = 768
    HD_DIM = 10000
    num_classes = 16

    # Loading the data
    arrays = np.load('/root/main/ScaLR/debug/nuscenes/SoA_results.npy')
    features = np.load('/root/main/ScaLR/debug/nuscenes/SoA_features.npy')
    labels = np.load('/root/main/ScaLR/debug/nuscenes/SoA_labels.npy')
    num_voxels = np.load('/root/main/ScaLR/debug/nuscenes/num_voxels.npy')

    #print("SOA results\n")
    #test_soa(arrays, labels, num_voxels, device)

    model = HD_Model(INPUT_DIM, HD_DIM, num_classes, device)

    model.train(features, labels, num_voxels)
    model.test_hd(features, labels, num_voxels)
    model.retrain(features, labels, num_voxels)
    model.test_hd(features, labels, num_voxels)