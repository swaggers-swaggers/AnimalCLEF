"""Image-trained DINOv3: spatial/CLS descriptor with low-rank backbone updates."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
import timm
from safetensors.torch import load_file
from settings import CV


class LowRank(nn.Module):
    def __init__(self, shape, rank=16):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, shape[1]))
        self.b = nn.Parameter(torch.zeros(shape[0], rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
    def forward(self, weight): return weight + self.b @ self.a


class SpatialDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=False, img_size=384, num_classes=0)
        self.backbone.load_state_dict(load_file(str(CV / 'animal_reid_v3/models/dinov3/model.safetensors')), strict=True)
        self.backbone.requires_grad_(False)
        for block in self.backbone.blocks[-8:]:
            for module in [block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2]:
                parametrize.register_parametrization(module, 'weight', LowRank(module.weight.shape))
            block.norm1.requires_grad_(True); block.norm2.requires_grad_(True)
        self.backbone.norm.requires_grad_(True)
        self.pool_power = nn.Parameter(torch.tensor(1.))
        self.backbone.set_grad_checkpointing(True)

    def forward(self, x):
        tokens = self.backbone.forward_features(x).float()
        patches = tokens[:, self.backbone.num_prefix_tokens:]
        p = self.pool_power.clamp(1., 3.)
        signed = (patches.sign() * patches.abs().clamp_min(1e-6).pow(p)).mean(1)
        pooled = signed.sign() * signed.abs().clamp_min(1e-6).pow(1 / p)
        return F.normalize(torch.cat([F.normalize(tokens[:, 0], dim=1), F.normalize(pooled, dim=1)], dim=1), dim=1)

    def adapter_state(self):
        names = set(n for n, p in self.named_parameters() if p.requires_grad)
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def load_adapter(self, state):
        expected = set(n for n, p in self.named_parameters() if p.requires_grad)
        assert set(state) == expected, 'Adapter architecture mismatch'
        self.load_state_dict(state, strict=False)


def queue_contrastive(z, y, memory, memory_y):
    candidates = torch.cat([z, memory.detach()], 0)
    labels = torch.cat([y, memory_y], 0)
    sim = z @ candidates.T / .08
    eye = torch.zeros_like(sim, dtype=torch.bool)
    eye[:, :len(z)] = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (y[:, None] == labels[None, :]) & ~eye
    logp = F.log_softmax(sim.masked_fill(eye, -1e4), dim=1)
    return -((logp * positives).sum(1) / positives.sum(1).clamp_min(1)).mean()
