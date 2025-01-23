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

        samples = (samples - mean) / (std + 1e-8)

        return samples
    
    def quantize_integer_to_nbit(self, tensor, n_bits):
        """
        Quantizes an integer tensor to a specified n-bit range.
        
        Args:
            tensor (torch.Tensor): The input tensor of integers.
            n_bits (int): The number of bits to represent the quantized range.

        Returns:
            torch.Tensor: The quantized tensor with values in the n-bit range.
        """
        # Define the target range based on n_bits
        target_min = -(2 ** (n_bits - 1))       # Minimum value for signed n-bit
        target_max = (2 ** (n_bits - 1)) - 1   # Maximum value for signed n-bit

        # Determine the source range from the input tensor
        source_min = -10000 #torch.min(tensor).item()
        source_max = 10000 #torch.max(tensor).item()

        # Step 1: Calculate scale factor
        scale = (source_max - source_min) / (target_max - target_min)

        # Step 2: Rescale and shift
        rescaled = (tensor - source_min) / scale + target_min

        # Step 3: Round to nearest integer and clip to target range
        quantized = torch.clamp(torch.round(rescaled), target_min, target_max)

        return quantized.int()  # Return as integer tensor

    def train(self, features, labels, num_voxels):

        """ Initial training pass """

        assert len(features) == len(labels)

        print("\nTrain First\n")

        for i in tqdm(range(len(features)), desc="1st Training:"):
            first_sample = features[i][:,:int(num_voxels[i])].to(self.device)
            first_sample = torch.transpose(first_sample, 0, 1)
            first_label = labels[i][:int(num_voxels[i])].to(torch.int32).to(self.device)

            first_sample = self.normalize(first_sample) # Z1 score seems to work
                
            # HD training

            samples_hv = self.encode(first_sample).to(torch.int32)

            ### Class Imbalance

            """samples_per_class = torch.bincount(first_label)
            samples_dif_0 = samples_per_class[samples_per_class != 0]
            classes_available = samples_dif_0.shape[0]
            weight_for_class_i = first_label.shape[0] / ( samples_dif_0 * classes_available)

            c = 0
            for real_c in range(self.num_classes):
                if samples_per_class[real_c] > 0:
                    #samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
                    here = first_label == real_c
                    self.model.add(samples_hv[here], first_label[here], lr=weight_for_class_i[c])
                    c += 1"""
            
            #### Original ####
            #temp = torch.zeros(self.num_classes, self.hd_dim, dtype=torch.int32).to(self.device)
            #temp.index_add_(0, first_label, samples_hv)
            #print("Min: ", torch.min(temp), "\nMax: ", torch.max(temp))
            #temp = temp
            # Add the 16 bit integer
            #self.model.weight = nn.Parameter(self.model.weight + temp, requires_grad=False) # Addition
            self.model.add(samples_hv, first_label)
            #print(self.model.weight)
            #x = input("Enter")

        # Normalizing works way better :)
        #self.model.normalize() # Min Max
        self.model.weight = nn.Parameter(torchhd.normalize(self.model.weight), requires_grad=False) # Binary

    def retrain(self, features, labels, num_voxels):
        
        """ Retrain with misclassified samples (also substract)"""
        
        for e in tqdm(range(10), desc="Epoch"):
            count = 0

            for i in range(len(features)):
                first_sample = features[i][:,:int(num_voxels[i])].to(self.device)
                first_sample = torch.transpose(first_sample, 0, 1)
                first_label = labels[i][:int(num_voxels[i])].to(torch.int32).to(self.device)

                first_sample = self.normalize(first_sample) # Z1 score seems to work

                #samples_per_class = torch.bincount(first_label)
                
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

                """
                for c in range(self.num_classes):
                    if samples_per_class[c] > 0:
                        #samples_hv = samples_hv.reshape((1,samples_hv.shape[0]))
                        here = first_label == c
                        self.model.weight.index_add_(0, first_label[here], samples_hv[here], alpha=weight_for_class_i[c])
                        self.model.weight.index_add_(0, pred_hd[here], samples_hv[here], alpha=-1*weight_for_class_i[c])
                """
                        
                #print(f"Misclassified for {i}: ", count)

                ## Original ###
                self.model.weight.index_add_(0, first_label, samples_hv)
                self.model.weight.index_add_(0, pred_hd, samples_hv, alpha=-1)

                ##### Try with int 16 #####
                #temp_1 = torch.zeros(self.num_classes, self.hd_dim, dtype=torch.int32).to(self.device)
                #temp_1.index_add_(0, first_label, samples_hv).to(torch.int16)
                #temp_2 = torch.zeros(self.num_classes, self.hd_dim, dtype=torch.int32).to(self.device)
                #temp_2.index_add_(0, pred_hd, samples_hv, alpha=-1).to(torch.int16)
                # Add the 16 bit integer
                #self.model.weight = nn.Parameter(self.model.weight + temp_1, requires_grad=False) # Addition
                #self.model.weight = nn.Parameter(self.model.weight + temp_2, requires_grad=False) # Addition

            # If you want to test for each sample
            #print(self.model.weight) # Int it is I think...
            #self.model.weight = nn.Parameter(torch.clamp(self.model.weight, min=-128, max=127).to(torch.int8), requires_grad=False)
            #print("Min model: ", torch.min(self.model.weight), "\nMax model: ", torch.max(self.model.weight))
            #print(self.model.weight)
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
    miou = MulticlassJaccardIndex(num_classes=19, average=None).to(device)
    final_shape = int(torch.sum(num_voxels))
    final_labels = torch.empty((final_shape), device=device)
    final_pred = torch.empty((final_shape), device=device)
    
    start_idx = 0
    for i in tqdm(range(len(results)), desc="Testing SoA"):
        shape_sample = int(num_voxels[i])
        first_sample = results[i][:shape_sample].to(device)
        first_label = labels[i][:shape_sample].to(torch.int64)
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
    num_classes = 19

    # Loading the data
    arrays = torch.load('/root/main/ScaLR/debug/semantic_kitti/soa_train_semkitti.pt')
    features = torch.load('/root/main/ScaLR/debug/semantic_kitti/feat_train_semkitti.pt')
    labels = torch.load('/root/main/ScaLR/debug/semantic_kitti/labels_train_semkitti.pt')
    num_voxels = torch.load('/root/main/ScaLR/debug/semantic_kitti/voxels_train_semkitti.pt')

    print("SOA results\n")
    test_soa(arrays, labels, num_voxels, device)

    model = HD_Model(INPUT_DIM, HD_DIM, num_classes, device)

    model.train(features, labels, num_voxels)
    model.test_hd(features, labels, num_voxels)
    model.retrain(features, labels, num_voxels)
    model.test_hd(features, labels, num_voxels)
