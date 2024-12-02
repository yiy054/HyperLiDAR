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
import wandb

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
        self.num_classes = num_classes

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

        for i in tqdm(range(len(features)), desc="1st Training:"):
            first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(self.device)
            first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int64).to(self.device)

            first_sample = self.normalize(first_sample) # Z1 score seems to work

            #for vox in range(len(first_sample)):
                
            #    label = first_label[vox]
            #    if vox % 5000 == 0:
            #        print(f"Sample {i}: Voxel {vox}")
                
            # HD training

            samples_hv = self.encode(first_sample)

            # Weights of each class
            
            samples_per_class = torch.bincount(first_label)
            samples_dif_0 = samples_per_class[samples_per_class != 0]
            classes_available = samples_dif_0.shape[0]
            weight_for_class_i = first_label.shape[0] / ( samples_dif_0 * classes_available)
            #print(weight_for_class_i.shape)
            #print(weight_for_class_i)
            weight_for_class_i = nn.functional.normalize(weight_for_class_i)

            #print("Labels")
            #print(samples_per_class)
            #print("Weights")
            #print(weight_for_class_i)

            ### Class Imbalance

            for c in range(self.num_classes):
                if samples_per_class[c] > 0:
                    #samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
                    here = first_label == c
                    self.model.add(samples_hv[here], first_label[here], lr=weight_for_class_i[c])
            
            #### Original ####
            #self.model.add(samples_hv, first_label)

            #### Normalize over inverse of the count #######

            #inverse_weights = 1.0 / (samples_per_class + 1.0)
    
            # Normalize the weights to sum to 1
            #normalized_weights = inverse_weights / torch.sum(inverse_weights)
            #print(normalized_weights)

            #for c in range(self.num_classes):
            #    if samples_per_class[c] > 0:
            #        #samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
            #        here = first_label == c
            #        self.model.add(samples_hv[here], first_label[here], lr=normalized_weights[c])


    def retrain(self, features, labels, num_voxels):
        
        """ Retrain with misclassified samples (also substract)"""
        
        for e in tqdm(range(10), desc="Epoch"):
            count = 0

            for i in range(len(features)):
                first_sample = torch.Tensor(features[i][:int(num_voxels[i])]).to(self.device)
                first_label = torch.Tensor(labels[i][:int(num_voxels[i])]).to(torch.int64).to(self.device)

                first_sample = self.normalize(first_sample) # Z1 score seems to work

                samples_per_class = torch.bincount(first_label)
                
                ##### Like loss for NN #########
                #weight_for_class_i = first_label.shape[0] / (( samples_per_class * num_classes) + 1e-6)
                
                ##### Inverse weights ####
                #inverse_weights = 1.0 / (samples_per_class + 1.0)
    
                # Normalize the weights to sum to 1
                #normalized_weights = inverse_weights / torch.sum(inverse_weights)

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

                #for c in range(self.num_classes):
                #    if samples_per_class[c] > 0:
                #        #samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
                #        here = first_label == c
                #        self.model.weight.index_add_(0, first_label[here], samples_hv[here], alpha=inverse_weights[c])
                #        self.model.weight.index_add_(0, pred_hd[here], samples_hv[here], alpha=-1*inverse_weights[c])

                #print(f"Misclassified for {i}: ", count)

                ## Original ###

                self.model.weight.index_add_(0, first_label, samples_hv)
                self.model.weight.index_add_(0, pred_hd, samples_hv, alpha=-1)

            # If you want to test for each sample
            self.test_hd(features, labels, num_voxels)

    def test_hd(self, features, labels, num_voxels, epoch=0):

        """ Testing over all the samples in all the scans given """

        assert len(features) == len(labels)
        
        # Metric
        miou = MulticlassJaccardIndex(num_classes=16, average=None).to(self.device)
        final_shape = int(np.sum(num_voxels))
        final_labels = torch.empty((final_shape), device=self.device)
        final_pred = torch.empty((final_shape), device=self.device)
        
        start_idx = 0
        for i in tqdm(range(len(features)), desc="Testing"):
            shape_sample = int(num_voxels[i])
            first_sample = torch.Tensor(features[i][:shape_sample]).to(self.device)
            first_label = torch.Tensor(labels[i][:shape_sample]).to(torch.int64)
            final_labels[start_idx:start_idx+shape_sample] = first_label

            first_sample = self.normalize(first_sample) # Z1 score seems to work

            # HD inference
            samples_hv = self.encode(first_sample)
            pred_hd = self.model(samples_hv, dot=True).argmax(1).data
            final_pred[start_idx:start_idx+shape_sample] = pred_hd

            start_idx += shape_sample

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

        #cm = confusion_matrix(pred_hd, first_label, labels=torch.Tensor(range(0,15)))
        #print("Confusion matrix \n")
        #print(cm)

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

    wandb.login(key="9487c04b8eff0c16cac4e785f2b57c3a475767d3")

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
    arrays = np.load('/root/main/ScaLR/debug/SoA_results.npy')
    features = np.load('/root/main/ScaLR/debug/SoA_features.npy')
    labels = np.load('/root/main/ScaLR/debug/SoA_labels.npy')
    num_voxels = np.load('/root/main/ScaLR/debug/num_voxels.npy')

    #print("SOA results\n")
    #test_soa(arrays, labels, num_voxels, device)

    model = HD_Model(INPUT_DIM, HD_DIM, num_classes, device)

    model.train(features, labels, num_voxels)
    model.test_hd(features, labels, num_voxels)
    model.retrain(features, labels, num_voxels)
    model.test_hd(features, labels, num_voxels)
