import os
from matplotlib import pyplot as plt
import numpy as np
from torch.utils import data
import yaml

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, 
                 sem_color_dict,
                 imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None, percentLabels=1):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))
           
        self.percentLabels=percentLabels
        self.sem_color_dict = sem_color_dict

        max_sem_key = 0
        for key, data in sem_color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for key, value in sem_color_dict.items():
            self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))

        # generate bitmask
        mask = np.ones(len(raw_data), dtype=bool)
        mask[:int(len(mask)*(1-self.percentLabels))] = False
        np.random.shuffle(mask)

        # put in attribute
        points = raw_data[:, 0:3]    # get xyz
        remissions = raw_data[:, 3]  # get remission
        label = annotated_data.astype(np.uint32)
        viridis_color, sem_label_color = self.get_colors(points, label)
            
        data_tuple = (points[mask, ...], remissions[mask, ...], label[mask, ...], viridis_color[mask, ...], sem_label_color[mask, ...])
        return data_tuple
    
    def get_colors(self, points, label):
        # plot scan
        power = 16
        # print()
        range_data = np.linalg.norm(points, 2, axis=1)
        #print(range_data.max(), range_data.min())
        range_data = range_data**(1 / power)
        #print(range_data.max(), range_data.min())
        viridis_range = ((range_data - range_data.min()) /
                        (range_data.max() - range_data.min()) *
                        255).astype(np.uint8)
        viridis_map = self.get_mpl_colormap("viridis")
        viridis_color = viridis_map[viridis_range]

        sem_label = label & 0xFFFF  # semantic label in lower half
        self.sem_label_color = self.sem_color_lut[sem_label]
        sem_label_color = self.sem_label_color.reshape((-1, 3))

        return viridis_color, sem_label_color

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0