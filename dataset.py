import numpy as np
import torch
import torch.utils.data as data

class TrackDataset(data.Dataset):
    def __init__(self, track_list, num_istances, num_labels):

        num_total = num_istances + num_labels
        self.index = []
        self.istances = []
        self.labels = []
        self.video = []
        self.vehicles = []

        for track in track_list:
            video = track.drive_name[-9:-5]
            vehicle = track.instance
            len_track = len(track.points)
            for count in range(0,len_track,1):
                if len_track - count > num_total:
                    temp_istance = track.points[count:count + num_istances].copy()
                    temp_label = track.points[count + num_istances:count + num_total].copy()

                    origin = temp_istance[-1]
                    st = temp_istance - origin
                    fu = temp_label - origin

                    self.index.append(count)
                    self.istances.append(st)
                    self.labels.append(fu)
                    self.video.append(video)
                    self.vehicles.append(vehicle)

        self.index = np.array(self.index)
        self.istances = torch.FloatTensor(self.istances)
        self.labels = torch.FloatTensor(self.labels)
        self.video = np.array(self.video)
        self.vehicles = np.array(self.vehicles)

    def __getitem__(self, idx):
        return self.index[idx], self.istances[idx], self.labels[idx], self.video[idx], self.vehicles[idx]

    def __len__(self):
        return len(self.istances)

