from auxiliary.ply_utils import read_ply, write_ply
import random
import torch

class Loader_Data():

    def __init__(self, cfg, device) -> None:
        self.cfg = cfg
        self.device = device

        if cfg.target == "paris":
            self.load = self.load_ply
    
    def load_ply(self, path):
        scan = read_ply('/root/main/dataset/training_10_classes/Lille1_1.ply')
        data = scan[['x', 'y', 'z']]
        label = scan['class']
        print("The min and max labels are: ", min(label), max(label))
        print("Number of points:", len(data))
        sample_size = int(len(data) * 0.5)

        # Randomly sample 75% of the tuple

        data_train = data[:sample_size]
        labels_train = label[:sample_size]
        data_test = data[sample_size:]
        labels_test = label[sample_size:]

        data_train_x = torch.FloatTensor([data_train['x'], data_train['y'], data_train['z']])
        data_train_x.to(self.device)
        labels_train_x = torch.FloatTensor([labels_train])
        labels_train_x.to(self.device)
        labels_train_x = labels_train_x[0]
        data_train_xt = torch.transpose(data_train_x, 0, 1)

        data_test_x = torch.FloatTensor([data_test['x'], data_test['y'], data_test['z']])
        data_test_x.to(self.device)
        labels_test_x = torch.FloatTensor([labels_test])
        labels_test_x.to(self.device)
        labels_test_x = labels_test_x[0]
        data_test_xt = torch.transpose(data_test_x, 0, 1)
        
        return data_train_xt, labels_train_x, data_test_xt, labels_test_x

