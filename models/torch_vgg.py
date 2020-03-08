import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

_, nn = try_import_torch()


class TorchVGG(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.features = self.extract_feature()
        self.hidden = nn.Sequential(
            nn.Linear(132, 64),
            nn.ReLU(True),
        )
        self.classifier = nn.Linear(64, num_outputs)
        self.value_branch = nn.Linear(64, 1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        board = input_dict['obs']['board']
        bomb_blast_strength = input_dict['obs']['bomb_blast_strength']
        bomb_life = input_dict['obs']['bomb_life']
        flame_life = input_dict['obs']['flame_life']
        position = input_dict['obs']['position']
        ammo = input_dict['obs']['ammo']
        blast_strength = input_dict['obs']['blast_strength']

        _inputs = torch.stack([board, bomb_blast_strength, bomb_life, flame_life], dim=0).reshape(-1, 4, board.shape[1],
                                                                                                  board.shape[1])
        _attributes = torch.cat([position, ammo, blast_strength], dim=1).reshape(-1, 4)
        features = self.features(_inputs).reshape(-1, 128)
        features = torch.cat([features, _attributes], dim=1)  # 1x132
        hidden = self.hidden(features)
        logits = self.classifier(hidden)
        self._cur_value = torch.squeeze(self.value_branch(hidden), 1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def extract_feature(self):
        layers = [
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        return nn.Sequential(*layers)
