from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2 as cv
import os

from sgan.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader

def img_loader(args, path):
    #/home/lyg/.../zara
    H = np.genfromtxt(os.path.join(path, 'H.txt'),
                    delimiter='  ',
                    unpack=True).transpose()

    print(H)

    grid_map = {}
    grid_map['Resolution'] = 0.1

    #map_center = grid_map['Size'] / 2  # hack for my dataset

    # Extract static obstacles
    obst_threshold = 200
    static_obst_img = cv.imread(os.path.join(path, "map.png"), 0)
    h = static_obst_img.shape[0]
    w = static_obst_img.shape[1]
    print(h, w)
    # grid_map['Size'] = np.dot(H, np.array([[h], [w], [1]]))
    # print(grid_map['Size'])
    grid_map['Map'] = np.zeros((h,w))

    obstacles = np.zeros([0, 3])
    for xx in range(static_obst_img.shape[0]):
        for yy in range(static_obst_img.shape[1]):
            if static_obst_img[xx, yy] > obst_threshold:
                obstacles = np.append(obstacles,
                                    np.dot(H, np.array([[xx], [yy], [1]])).transpose(),
                                    axis=0)
    Hinv = np.linalg.inv(H)

    # Compute obstacles in 2D
    obstacles_2d = np.zeros([obstacles.shape[0], 2])
    obstacles_2d[:, 0] = obstacles[:, 0] / obstacles[:, 2]
    obstacles_2d[:, 1] = obstacles[:, 1] / obstacles[:, 2]

    # Get obstacle idx on map
    obst_idx = []
    for obst_ii in range(obstacles_2d.shape[0]):
        idx = np.dot(Hinv, np.array([[obstacles_2d[obst_ii, 0]], [obstacles_2d[obst_ii, 1]], [1]])).transpose()
        obst_idx.append((int(idx[0,0]),int(idx[0,1])))
        grid_map['Map'][obst_idx[-1]] = 1


    print("Obstacle shape: ", grid_map['Map'].shape)
    print("num_obst: ", np.sum(grid_map['Map']))
    gmap = cv.resize(grid_map['Map'], (224, 224))
    return torch.from_numpy(gmap).type(torch.float)


