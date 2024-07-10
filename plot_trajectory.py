import os
import time
import numpy as np
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_singleLayer import model_singleLayer
from model_multiLayer import model_multiLayer
import dataset
from random import randint
import datetime
from tensorboardX import SummaryWriter
import parse_trajectories
import pdb
import json
import dataset_invariance
import tqdm

dim_clip = 180
tracks = json.load(open('kitti_dataset.json'))
data_train = dataset_invariance.TrackDataset(tracks,
                                             len_past=20,
                                             len_future=40,
                                             train=True,
                                             dim_clip=dim_clip)

train_loader = DataLoader(data_train,
                          batch_size=2,
                          num_workers=1,
                          shuffle=True)

data_test = dataset_invariance.TrackDataset(tracks,
                                            len_past=20,
                                            len_future=40,
                                            train=False,
                                            dim_clip=dim_clip)

test_loader = DataLoader(data_test,
                         batch_size=32,
                         num_workers=1,
                         shuffle=False)

for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene,
           scene_one_hot) in enumerate(tqdm.tqdm(test_loader)):
    x = 2
# # draw_track(self, past, future, index_tracklet, path)
# # draw_scene(self, scene_track, index_tracklet, path)
# # draw_track_in_scene(self, story, scene_track, index_tracklet, future=None, path='')
