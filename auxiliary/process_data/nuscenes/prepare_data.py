import argparse
import os
import numpy as np
from sklearn.decomposition import PCA
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_io import load_bin_file

parser = argparse.ArgumentParser()
parser.add_argument("--rootdir", type=str, required=True)
parser.add_argument("--destdir", type=str, required=True)
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

nusc = NuScenes(version='v1.0-mini', dataroot=args.rootdir, verbose=True)

# create the directory
train_filenames = []
test_filenames = []

if args.test:
    filenames = test_filenames
    save_dir = os.path.join(args.destdir,"test_pointclouds")
else:
    filenames = train_filenames
    save_dir = os.path.join(args.destdir,"train_pointclouds")
os.makedirs(save_dir, exist_ok=True)


for sc in nusc.scene:
    file = []
    labels = []
    sample = nusc.get('sample',sc['first_sample_token'])
    sample_data = sample['data']['LIDAR_TOP']
    record = nusc.get('sample_data', sample_data)
    pcl_path = os.path.join(nusc.dataroot, record['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    points = pc.points
    label_file = os.path.join(nusc.dataroot, nusc.get('lidarseg',sample_data)['filename'])
    points_label = load_bin_file(label_file, type='lidarseg')

    print(points_label.shape)
    x = 0 
    y = 0
    z = 0
    reflectance = 0
    if not args.test:
        label = 0
    
    if args.test:
        pts = np.concatenate([
                np.expand_dims(x,1),
                np.expand_dims(y,1),
                np.expand_dims(z,1),
                np.expand_dims(reflectance,1),
                ], axis=1).astype(np.float32)

        np.save(os.path.join(save_dir, os.path.splitext(filename)[0]), pts)
        
    else:
        pts = np.concatenate([
                np.expand_dims(x,1),
                np.expand_dims(y,1),
                np.expand_dims(z,1),
                np.expand_dims(reflectance,1),
                np.expand_dims(label,1),
                ], axis=1).astype(np.float32)

        pca = PCA(n_components=1)
        pca.fit(pts[::10,:2])
        pts_new = pca.transform(pts[:,:2])
        hist, edges = np.histogram(pts_new, pts_new.shape[0]// 2500000)

        count = 0

        for i in range(1,edges.shape[0]):
            mask = np.logical_and(pts_new<=edges[i], pts_new>edges[i-1])[:,0]
            np.save(os.path.join(save_dir, os.path.splitext(filename)[0]+f"_{count}"), pts[mask])
            count+=1


        hist, edges = np.histogram(pts_new, pts_new.shape[0]// 2500000 -2, range=[(edges[1]+edges[0])//2,(edges[-1]+edges[-2])//2])

        for i in range(1,edges.shape[0]):
            mask = np.logical_and(pts_new<=edges[i], pts_new>edges[i-1])[:,0]
            np.save(os.path.join(save_dir, os.path.splitext(filename)[0]+f"_{count}"), pts[mask])
            count+=1