"""Audited local architectures; Hugging Face remote Python is never executed.

MiewID architecture matches the published conservationxlabs/miewid-msv3 definition.
MegaDescriptor uses timm's Swin-L code; the released weights are CC BY-NC 4.0.
"""
import json
import timm
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from common import ROOT

class GeM(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([3.0]))

    def forward(self, x):
        return x.clamp_min(1e-6).pow(self.p).mean((-2, -1), keepdim=True).pow(1 / self.p)

class MiewID(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnetv2_rw_m', pretrained=False, num_classes=0)
        self.backbone.global_pool = GeM()
        self.bn = nn.BatchNorm1d(2152)
        self.final = nn.Linear(2152, 10)

    def forward(self, x):
        return self.bn(self.backbone(x).flatten(1))

def load_global(name):
    if name == 'miewid':
        model = MiewID()
        model.load_state_dict(load_file(str(ROOT / 'models' / 'miewid.safetensors')), strict=True)
        return model.eval().cuda(), 440
    if name == 'mega':
        model = timm.create_model('swin_large_patch4_window12_384', num_classes=0, pretrained=False)
        state = torch.load(ROOT / 'models' / 'megadescriptor.bin', map_location='cpu', weights_only=True)
        # timm's checkpoint filter handles original Swin key names and layout.
        from timm.models.swin_transformer import checkpoint_filter_fn
        model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
        return model.eval().cuda(), 384
    raise ValueError(name)
