import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class ProjectBlock(nn.Module):
    def __init__(self, mm_hidden_size,hidden_size):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(mm_hidden_size*3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )



        self.img_high_q_proj = nn.Linear(mm_hidden_size,mm_hidden_size)
        self.img_high_k_proj = nn.Linear(mm_hidden_size,mm_hidden_size)
        self.img_high_v_proj = nn.Linear(mm_hidden_size,mm_hidden_size)

        self.img_low_q_proj = nn.Linear(mm_hidden_size,mm_hidden_size)
        self.img_low_k_proj = nn.Linear(mm_hidden_size,mm_hidden_size)
        self.img_low_v_proj = nn.Linear(mm_hidden_size,mm_hidden_size)

        self.q_proj = nn.Linear(mm_hidden_size,mm_hidden_size)
        self.k_proj = nn.Linear(mm_hidden_size,mm_hidden_size)
        self.v_proj = nn.Linear(mm_hidden_size,mm_hidden_size)

        self.txt_q_proj = nn.Linear(mm_hidden_size,mm_hidden_size)
        self.txt_k_proj = nn.Linear(mm_hidden_size,mm_hidden_size)
        self.txt_v_proj = nn.Linear(mm_hidden_size,mm_hidden_size)


    def forward(self, x):
        return self.proj(x)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    # mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    # if mlp_gelu_match:
    #     mlp_depth = int(mlp_gelu_match.group(1))
    #     modules = [nn.Linear(config.mm_hidden_size*3, config.hidden_size)]
    #     for _ in range(1, mlp_depth):
    #         modules.append(nn.GELU())
    #         modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    #     return nn.Sequential(*modules)


    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        return ProjectBlock(config.mm_hidden_size, config.hidden_size)    


    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
