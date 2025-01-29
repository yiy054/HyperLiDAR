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
        self.batch_size = 10000

    def normalize(self, samples):

        """ Normalize with Z-score"""

        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)

        #print("Mean in range: ", min(mean), " ", max(mean))
        #print("Std in range: ", min(std), " ", max(std))

        samples = (samples - mean) / (std + 1e-8)

        """Min-max normalization"""

        # Compute the minimum and maximum of the tensor
        #min_val = samples.min(axis=0).values
        #max_val = samples.max(axis=0).values

        # Perform Min-Max normalization
        #normalized_tensor = (samples - min_val) / (max_val - min_val)
        #samples = normalized_tensor #* (max_range - min_range) + min_range

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

    def train(self, features, labels, num_voxels, points):

        """ Initial training pass """

        assert len(features) == len(labels)

        print("\nTrain First\n")

        batch = 10000

        for i in tqdm(range(len(features)), desc="1st Training:"):
            #print("Min and max of this sample \n Min: ", torch.min(points[i], axis=1).values, "\nMax: ", torch.max(points[i], axis=1).values)
            for b in range(0,int(num_voxels[i]), batch):
                end = min(b + batch, int(num_voxels[i]))  # Ensure we don't exceed num_voxels[i]
                points_here = points[i][:,b:end]
                # Assuming points_here is a 2D tensor with shape (N, 3)

                first_sample = features[i][:,b:end].to(self.device)
                first_sample = torch.transpose(first_sample, 0, 1)
                first_label = labels[i][b:end].to(torch.int32).to(self.device)
                first_sample = self.normalize(first_sample) # Z1 score seems to work
                    
                # HD training
                samples_hv = self.encode(first_sample).to(torch.int32)

                if i==0 and b==0:
                    # Apply t-SNE to reduce dimensions to 2D
                    tsne = TSNE(n_components=2, perplexity=10)
                    features_2d = tsne.fit_transform(samples_hv.cpu())

                    # Plot the t-SNE result
                    plt.figure(figsize=(10, 8))
                    scatter = sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=first_label.cpu(), palette="tab10", alpha=0.7)
                    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.xlabel("t-SNE Component 1")
                    plt.ylabel("t-SNE Component 2")
                    plt.title("t-SNE Visualization of Features")

                    # Save the plot
                    plt.savefig("tsne_plot_sem_kitti.png", dpi=300, bbox_inches="tight")

                    # Show the plot (optional)
                    plt.show()

                ### Class Imbalance
                self.model.add(samples_hv, first_label)

        # Normalizing works way better :)
        #self.model.normalize() # Min Max
        self.model.weight = nn.Parameter(torchhd.normalize(self.model.weight), requires_grad=False) # Binary

    def retrain(self, features, labels, num_voxels, points, features_test, labels_test, num_voxel_test, points_test):
        
        """ Retrain with misclassified samples (also substract)"""

        batch = 20000
        
        for e in tqdm(range(10), desc="Epoch"):
            count = 0

            for i in range(len(features)):
                for b in range(0,int(num_voxels[i]), batch):
                    end = min(b + batch, int(num_voxels[i]))  # Ensure we don't exceed num_voxels[i]
                    first_sample = features[i][:,b:end].to(self.device)
                    first_sample = torch.transpose(first_sample, 0, 1)
                    first_label = labels[i][b:end].to(torch.int32).to(self.device)
                    print(torch.bincount(first_label))
                    first_sample = self.normalize(first_sample)

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

                    ## Original ###
                    self.model.weight.index_add_(0, first_label, samples_hv)
                    self.model.weight.index_add_(0, pred_hd, samples_hv, alpha=-1)

            # If you want to test for each sample
            self.test_hd(features_test, labels_test, num_voxel_test, points_test, epoch=e+1)

    def test_hd(self, features, labels, num_voxels, points, epoch=0):

        """ Testing over all the samples in all the scans given """

        assert len(features) == len(labels)
        
        # Metric
        miou = MulticlassJaccardIndex(num_classes=19, average=None).to(self.device)
        final_shape = int(torch.sum(num_voxels))
        final_labels = torch.empty((final_shape), device=self.device)
        final_pred = torch.empty((final_shape), device=self.device)
        
        start_idx = 0
        batch = 20000
        for i in tqdm(range(len(features)), desc="Testing"):
            for b in range(0,int(num_voxels[i]), batch):
                end = min(b + batch, int(num_voxels[i]))  # Ensure we don't exceed num_voxels[i]
                first_sample = features[i][:,b:end].to(self.device)
                first_sample = torch.transpose(first_sample, 0, 1)
                first_label = labels[i][b:end].to(torch.int64)
                final_labels[start_idx:start_idx+end-b] = first_label

                first_sample = self.normalize(first_sample) # Z1 score seems to work

                # HD inference
                samples_hv = self.encode(first_sample)
                pred_hd = self.model(samples_hv, dot=True).argmax(1).data
                final_pred[start_idx:start_idx+end-b] = pred_hd

                start_idx += end-b

        print("================================")

        #print('pred_ts', pred_ts)
        print('label', final_labels, "\tShape: ", final_labels.shape)
        print('pred_hd', final_pred, "\tShape: ", final_pred.shape)
        accuracy = miou(final_pred, final_labels)
        avg_acc = torch.mean(accuracy)
        print(f'accuracy: {accuracy}')
        print(f'avg acc: {avg_acc}')

        #log_data = {f"Training class_{i}_IoU": c for i, c in enumerate(accuracy)}
        #log_data["Retraining epoch"] = avg_acc
        #wandb.log(log_data)

        # Compute the confusion matrix
        """cm = confusion_matrix(final_labels.cpu().numpy(), final_pred.cpu().numpy(), labels=torch.arange(18).numpy())

        # Plot the confusion matrix
        plt.figure(figsize=(16, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(18), yticklabels=range(18))
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for Epoch {epoch}")

        # Save the figure
        plt.savefig(f"confusion_matrix_{epoch}.png", dpi=300, bbox_inches="tight")"""

        print("================================")

def test_soa(results, labels, num_voxels, points, device):
    assert len(results) == len(labels)
        
    # Metric
    miou = MulticlassJaccardIndex(num_classes=19, average=None).to(device)
    print(num_voxels)
    final_shape = int(torch.sum(num_voxels))
    print(final_shape)
    final_labels = torch.empty((final_shape), device=device)
    final_pred = torch.empty((final_shape), device=device)
    
    start_idx = 0
    for i in tqdm(range(len(results)), desc="Testing SoA"):
        shape_sample = int(num_voxels[i])
        first_sample = results[i][:shape_sample].to(device)
        first_label = labels[i][:shape_sample].to(torch.int64)
        final_labels[start_idx:start_idx+shape_sample] = first_label

        pred = first_sample#.max(1)[1]
        final_pred[start_idx:start_idx+shape_sample] = pred

        start_idx += shape_sample

    print("================================")

    print('label', final_labels, "\tShape: ", final_labels.shape)
    print('pred', final_pred, "\tShape: ", final_pred.shape)
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
    HD_DIM = 50000
    num_classes = 19

    # Loading the data
    arrays = torch.load('/root/main/ScaLR/debug/semantic_kitti/soa_train_semkitti.pt', weights_only="False")
    features = torch.load('/root/main/ScaLR/debug/semantic_kitti/feat_train_semkitti.pt', weights_only="False")
    labels = torch.load('/root/main/ScaLR/debug/semantic_kitti/labels_train_semkitti.pt', weights_only="False")
    num_voxels = torch.load('/root/main/ScaLR/debug/semantic_kitti/voxels_train_semkitti.pt', weights_only="False")
    points = torch.load('/root/main/ScaLR/debug/semantic_kitti/pts_train_semkitti.pt', weights_only="False")

    # Assuming points_here is a 2D tensor with shape (N, 3)
    #for i, vox in enumerate(num_voxels):
    #    points_here = points[i][:, :vox]
    #    points_here_idx = (
    #        (points_here[0, :] > -10) & (points_here[0, :] < 10) &
    #        (points_here[1, :] > -10) & (points_here[1, :] < 10) &
    #        (points_here[2, :] > -10) & (points_here[2, :] < 10)
    #    )
    #    new_shape = int(sum(points_here_idx))
    #    features[i][:, :new_shape] = features[i][:, :vox][:, points_here_idx]
        #[:, points_here_idx]
    #    labels[i][:new_shape] = labels[i][:vox][points_here_idx]
    #    num_voxels[i] = new_shape

    arrays_test = torch.load('/root/main/ScaLR/debug/semantic_kitti/soa_test_semkitti.pt', weights_only="False")
    features_test = torch.load('/root/main/ScaLR/debug/semantic_kitti/feat_test_semkitti.pt', weights_only="False")
    labels_test = torch.load('/root/main/ScaLR/debug/semantic_kitti/labels_test_semkitti.pt', weights_only="False")
    num_voxels_test = torch.load('/root/main/ScaLR/debug/semantic_kitti/voxels_test_semkitti.pt', weights_only="False")
    points_test = torch.load('/root/main/ScaLR/debug/semantic_kitti/pts_test_semkitti.pt', weights_only="False")

    # Assuming points_here is a 2D tensor with shape (N, 3)
    #for i, vox in enumerate(num_voxels_test):
    #    points_here = points_test[i][:, :vox]
    #    points_here_idx = (
    #        (points_here[0, :] > -10) & (points_here[0, :] < 10) &
    #        (points_here[1, :] > -10) & (points_here[1, :] < 10) &
    #        (points_here[2, :] > -10) & (points_here[2, :] < 10)
    #    )
    #    new_shape = int(sum(points_here_idx))
    #    features_test[i][:, :new_shape] = features_test[i][:, :vox][:, points_here_idx]
        #[:, points_here_idx]
    #   labels_test[i][:new_shape] = labels_test[i][:vox][points_here_idx]
    #    arrays_test[i][:new_shape] = arrays_test[i][:vox][points_here_idx]
    #    num_voxels_test[i] = new_shape#

    print("SOA results\n")
    test_soa(arrays_test, labels_test, num_voxels_test, points, device)

    model = HD_Model(INPUT_DIM, HD_DIM, num_classes, device)

    model.train(features, labels, num_voxels, points)
    model.test_hd(features_test, labels_test, num_voxels_test, points)
    model.retrain(features, labels, num_voxels, points, features_test, labels_test, num_voxels_test, points_test)
    model.test_hd(features_test, labels_test, num_voxels_test, points_test)
