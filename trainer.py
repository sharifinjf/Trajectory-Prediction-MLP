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


class Trainer():
    def __init__(self, config):

        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'test/' + self.name_test
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")

        self.dim_clip = 180
        tracks = json.load(open(config.dataset_file))
        self.data_train = dataset_invariance.TrackDataset(tracks,
                                                          len_past=config.past_len,
                                                          len_future=config.future_len,
                                                          train=True,
                                                          dim_clip=self.dim_clip)

        self.train_loader = DataLoader(self.data_train,
                                       batch_size=2,
                                       num_workers=1,
                                       shuffle=True)

        self.data_test = dataset_invariance.TrackDataset(tracks,
                                                         len_past=config.past_len,
                                                         len_future=config.future_len,
                                                         train=False,
                                                         dim_clip=self.dim_clip)

        self.test_loader = DataLoader(self.data_test,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False)

        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
        }
        self.max_epochs = config.max_epochs
        if config.model_non_linear:
            self.mem_n2n = model_multiLayer(self.settings)
        else:
            self.mem_n2n = model_singleLayer(self.settings)

        self.criterionLoss = nn.MSELoss()
        self.EuclDistance = nn.PairwiseDistance(p=2)
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.5)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config
        self.write_details()
        self.file.close()

        self.writer = SummaryWriter('runs-test/' + self.name_test)

    def write_details(self):
        self.file.write("non-linear model?: " + str(self.config.model_non_linear) + '\n')
        self.file.write("points of past track: " + str(self.config.past_len) + '\n')
        self.file.write("points of future track: " + str(self.config.future_len) + '\n')
        self.file.write("train size: " + str(len(self.data_train)) + '\n')
        self.file.write("test size: " + str(len(self.data_test)) + '\n')
        self.file.write("batch size: " + str(self.config.batch_size) + '\n')

    def draw_track(self, story, future, pred=None, index_tracklet=0, num_epoch=0, saveFig=False, path=''):
        config = self.config
        fig = plt.figure()
        story = story.cpu().numpy()
        future = future.view(int(self.settings["dim_feature_future"] / 2), 2).cpu().numpy()
        plt.plot(story[:, 0], story[:, 1], c='blue', marker='o', markersize=3)
        plt.plot(future[:, 0], future[:, 1], c='green', marker='o', markersize=3)
        if pred is not None:
            pred = pred.view(int(self.settings["dim_feature_future"] / 2), 2).cpu().numpy()
            plt.plot(pred[:, 0], pred[:, 1], c='red', linewidth=1, marker='o', markersize=3)
        plt.axis('equal')

        if saveFig:
            plt.savefig(path + str(index_tracklet.data) + '.png')

            ## save figure in tensorboard
            # buf = io.BytesIO()
            # plt.savefig(buf, format='jpeg')
            # buf.seek(0)
            # image = Image.open(buf)
            # image = ToTensor()(image).unsqueeze(0)
            # if train:
            #    self.writer.add_image('Image_train/track' + str(index_tracklet), image.squeeze(0), num_epoch)
            # else:
            #    self.writer.add_image('Image_test/track' + str(index_tracklet), image.squeeze(0), num_epoch)
        plt.close(fig)

    def fit(self):
        config = self.config
        for epoch in range(self.start_epoch, config.max_epochs):

            print('epoch: ' + str(epoch))
            loss = self._train_single_epoch(epoch)

            if (epoch + 1) % 10 == 0:
                dictMetrics_train = self.evaluate(self.train_loader, epoch + 1)
                dictMetrics_test = self.evaluate(self.test_loader, epoch + 1)

                for param_group in self.opt.param_groups:
                    self.writer.add_scalar('learning_rate', param_group["lr"], epoch)

                self.writer.add_scalar('accuracy_train/euclMean', dictMetrics_train['euclMean'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon01s', dictMetrics_train['horizon01s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon10s', dictMetrics_train['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon20s', dictMetrics_train['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon30s', dictMetrics_train['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon40s', dictMetrics_train['horizon40s'], epoch)
                self.writer.add_scalar('accuracy_train/mhd_error', dictMetrics_train['mhd_error'], epoch)
                self.writer.add_scalar('accuracy_train/ADE10)', dictMetrics_train['ADE10'], epoch)
                self.writer.add_scalar('accuracy_train/ADE20)', dictMetrics_train['ADE20'], epoch)
                self.writer.add_scalar('accuracy_train/ADE30)', dictMetrics_train['ADE30'], epoch)
                self.writer.add_scalar('accuracy_train/ADE40)', dictMetrics_train['ADE40'], epoch)

                self.writer.add_scalar('accuracy_test/euclMean', dictMetrics_test['euclMean'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon01s', dictMetrics_test['horizon01s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dictMetrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon20s', dictMetrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dictMetrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon40s', dictMetrics_test['horizon40s'], epoch)
                self.writer.add_scalar('accuracy_test/mhd_error', dictMetrics_test['mhd_error'], epoch)
                self.writer.add_scalar('accuracy_test/ADE10)', dictMetrics_test['ADE10'], epoch)
                self.writer.add_scalar('accuracy_test/ADE20)', dictMetrics_test['ADE20'], epoch)
                self.writer.add_scalar('accuracy_test/ADE30)', dictMetrics_test['ADE30'], epoch)
                self.writer.add_scalar('accuracy_test/ADE40)', dictMetrics_test['ADE40'], epoch)
                print('ADE10', dictMetrics_test['ADE10'], 'ADE20', dictMetrics_test['ADE20'], 'ADE30',
                      dictMetrics_test['ADE30'], 'ADE40', dictMetrics_test['ADE40'])
                print('FDE10', dictMetrics_test['horizon10s'], 'FDE20', dictMetrics_test['horizon20s'], 'FDE30',
                      dictMetrics_test['horizon30s'], 'FDE40', dictMetrics_test['horizon40s'])

                torch.save(self.mem_n2n, self.folder_test + 'model' + self.name_test)

            torch.save(self.mem_n2n, self.folder_test + 'model' + self.name_test)

    def load(self, directory):
        pass

    def evaluate(self, loader, epoch=0):

        euclMean = 0
        horizon01s = 0
        horizon10s = 0
        horizon20s = 0
        horizon30s = 0
        horizon40s = 0
        mhd_error = 0
        ADE10 = 0
        ADE20 = 0
        ADE30 = 0
        ADE40 = 0
        dictMetrics = {}
        listError = []

        for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene,
                   scene_one_hot) in enumerate(tqdm.tqdm(loader)):

            story = Variable(past)
            future = Variable(future)
            if self.config.cuda:
                story = story.cuda()
                future = future.cuda()
            pred = self.mem_n2n(story).data
            pred = pred.view(pred.shape[0], future.shape[1], 2)
            for i in range(len(story)):
                dist = self.EuclDistance(pred[i], future[i])
                euclMean += torch.mean(dist)
                listError.append(torch.mean(dist))

                if future.shape[1] > 30:
                    horizon01s += dist[0]
                    horizon10s += dist[9]
                    horizon20s += dist[19]
                    horizon30s += dist[29]
                    horizon40s += dist[39]
                    ADE10 += torch.mean(dist[0:9])
                    ADE20 += torch.mean(dist[0:19])
                    ADE30 += torch.mean(dist[0:29])
                    ADE40 += torch.mean(dist[0:39])

                (MHD, FHD, RHD) = self.ModHausdorffDist(pred[i].cpu().numpy(), future[i].cpu().numpy())

                mhd_error += FHD
                # if loader == self.test_loader and epoch == self.max_epochs:
                #     vid = videos[i]
                #     vec = vehicles[i]
                #     if not os.path.exists(self.folder_test + vid):
                #         os.makedirs(self.folder_test + vid)
                #     video_path = self.folder_test + vid + '/'
                #     if not os.path.exists(video_path + vec):
                #         os.makedirs(video_path + vec)
                #     vehicle_path = video_path + vec + '/'
                #     self.draw_track(story[i], future[i], pred[i], index_tracklet=index[i], num_epoch=epoch, saveFig=True,
                #                     path=vehicle_path)

        dictMetrics['euclMean'] = euclMean / len(loader.dataset)
        dictMetrics['horizon01s'] = horizon01s / len(loader.dataset)
        dictMetrics['horizon10s'] = horizon10s / len(loader.dataset)
        dictMetrics['horizon20s'] = horizon20s / len(loader.dataset)
        dictMetrics['horizon30s'] = horizon30s / len(loader.dataset)
        dictMetrics['horizon40s'] = horizon40s / len(loader.dataset)
        dictMetrics['ADE10'] = ADE10 / len(loader.dataset)
        dictMetrics['ADE20'] = ADE20 / len(loader.dataset)
        dictMetrics['ADE30'] = ADE30 / len(loader.dataset)
        dictMetrics['ADE40'] = ADE40 / len(loader.dataset)
        dictMetrics['mhd_error'] = mhd_error / len(loader.dataset)
        return dictMetrics

    def _train_single_epoch(self, epoch):
        config = self.config
        for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene,
                   scene_one_hot) in enumerate(tqdm.tqdm(self.train_loader)):
            self.iterations += 1
            story = Variable(past)
            future = Variable(future)
            if config.cuda:
                story = story.cuda()
                future = future.cuda()

            self.opt.zero_grad()
            output = self.mem_n2n(story)
            loss = self.criterionLoss(output, future.view(future.shape[0], -1))
            loss.backward()
            self.opt.step()
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)

        return loss.item()

    def ModHausdorffDist(self, A, B):
        # This function computes the Modified Hausdorff Distance (MHD) which is
        # proven to function better than the directed HD as per Dubuisson et al.
        # in the following work:
        #
        # M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
        # matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
        # http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
        #
        # The function computed the forward and reverse distances and outputs the
        # maximum/minimum of both.
        # Optionally, the function can return forward and reverse distance.
        #
        # Format for calling function:
        #
        # [MHD,FHD,RHD] = ModHausdorffDist(A,B);
        #
        # where
        # MHD = Modified Hausdorff Distance.
        # FHD = Forward Hausdorff Distance: minimum distance from all points of B
        #      to a point in A, averaged for all A
        # RHD = Reverse Hausdorff Distance: minimum distance from all points of A
        #      to a point in B, averaged for all B
        # A -> Point set 1, [row as observations, and col as dimensions]
        # B -> Point set 2, [row as observations, and col as dimensions]
        #
        # No. of samples of each point set may be different but the dimension of
        # the points must be the same.
        #
        # Edward DongBo Cui Stanford University; 06/17/2014

        # Find pairwise distance
        # pdb.set_trace()
        D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
        # Calculating the forward HD: mean(min(each col))
        FHD = np.mean(np.min(D_mat, axis=1))
        # Calculating the reverse HD: mean(min(each row))
        RHD = np.mean(np.min(D_mat, axis=0))
        # Calculating mhd
        MHD = np.max(np.array([FHD, RHD]))
        return (MHD, FHD, RHD)
