import torch.nn as nn
import json
import tinycudann as tcnn
from utils import mask


class motion_model(nn.Module):
    def __init__(self):
        super(motion_model, self).__init__()
        with open("config/config_medium.json") as config_file:
            config = json.load(config_file)

        self.flow_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=config["encoding"],
        )

        self.flow_net = tcnn.Network(
            n_input_dims=self.flow_encoding.n_output_dims,
            n_output_dims=2,
            network_config=config["network"],
        )

    def forward(self, tyx, sin_epoch):
        # tyx: [T*H*W, 3]
        flow_encoding = self.flow_encoding(tyx)  # [T*H*W, 2]
        flow_encoding = mask(flow_encoding, sin_epoch)
        flow = self.flow_net(flow_encoding)  # [T*H*W, 2]
        return flow
