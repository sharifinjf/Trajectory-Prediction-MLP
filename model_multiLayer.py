import torch
import torch.nn as nn

class model_multiLayer(nn.Module):
    def __init__(self, settings):
        super(model_multiLayer, self).__init__()

        #parameters
        self.use_cuda = settings["use_cuda"]
        self.dim_feature_tracklet = settings["dim_feature_tracklet"]
        self.dim_feature_future = settings["dim_feature_future"]

        #layers
        self.firstLayer = torch.nn.Linear(self.dim_feature_tracklet, 60)
        self.secondLayer = torch.nn.Linear(60, self.dim_feature_future)
        self.relu = nn.ReLU()

        # weight initialization: xavier
        torch.nn.init.xavier_uniform_(self.firstLayer.weight)
        torch.nn.init.xavier_uniform_(self.secondLayer.weight)
        torch.nn.init.zeros_(self.firstLayer.bias)
        torch.nn.init.zeros_(self.secondLayer.bias)

    def forward(self, story):

        dim_batch = story.size()[0]
        story = story.view(dim_batch, -1)
        hidden = self.relu(self.firstLayer(story))
        output = self.secondLayer(hidden)
        return output
