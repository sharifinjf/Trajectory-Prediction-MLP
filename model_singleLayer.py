import torch
import torch.nn as nn

class model_singleLayer(nn.Module):
    def __init__(self, settings):
        super(model_singleLayer, self).__init__()

        #parameters
        self.use_cuda = settings["use_cuda"]
        self.dim_feature_tracklet = settings["dim_feature_tracklet"]
        self.dim_feature_future = settings["dim_feature_future"]

        #layers
        self.oneLayer = torch.nn.Linear(self.dim_feature_tracklet, self.dim_feature_future)

        # weight initialization: Xavier
        torch.nn.init.xavier_uniform_(self.oneLayer.weight)
        torch.nn.init.zeros_(self.oneLayer.bias)


    def forward(self, story):

        dim_batch = story.size()[0]
        story = story.view(dim_batch, -1)
        output = self.oneLayer(story)
        return output




