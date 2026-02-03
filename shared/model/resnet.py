"""
resnet.py - A modified ResNet structure
We append extra channels to the first conv by some network surgery
"""

from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo

from torch.hub import load
import os
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

from einops import rearrange

def load_weights_add_extra_dim(target, source_state, extra_dim=1):
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            if k1 in source_state:
                tar_v = source_state[k1]

                if v1.shape != tar_v.shape:
                    # Init the new segmentation channel with zeros
                    # print(v1.shape, tar_v.shape)
                    c, _, w, h = v1.shape
                    pads = torch.zeros((c,extra_dim,w,h), device=tar_v.device)
                    nn.init.orthogonal_(pads)
                    tar_v = torch.cat([tar_v, pads], 1)

                new_dict[k1] = tar_v

    target.load_state_dict(new_dict)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), extra_dim=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3+extra_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

def resnet18(pretrained=True, extra_dim=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_dim)
    if pretrained:
        load_weights_add_extra_dim(model, model_zoo.load_url(model_urls['resnet18']), extra_dim)
    return model

def resnet50(pretrained=True, extra_dim=0):
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_dim)
    if pretrained:
        load_weights_add_extra_dim(model, model_zoo.load_url(model_urls['resnet50']), extra_dim)
    return model

DINO_BACKBONES = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14,
        'hub_repo': 'facebookresearch/dinov2',
        'num_prefix_tokens': 1,
        'source': 'torch_hub',
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14,
        'hub_repo': 'facebookresearch/dinov2',
        'num_prefix_tokens': 1,
        'source': 'torch_hub',
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14,
        'hub_repo': 'facebookresearch/dinov2',
        'num_prefix_tokens': 1,
        'source': 'torch_hub',
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14,
        'hub_repo': 'facebookresearch/dinov2',
        'num_prefix_tokens': 1,
        'source': 'torch_hub',
    },
    'dinov3_vits16':{
        'name':'dinov3_vits16',
        'embedding_size':384,
        'patch_size':16,
        'hub_repo': 'facebookresearch/dinov3',
        'num_prefix_tokens': 5,
        'source': 'torch_hub',
        'hf_id': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    },
    'dinov3_vitb16':{
        'name':'dinov3_vitb16',
        'embedding_size':768,
        'patch_size':16,
        'hub_repo': 'facebookresearch/dinov3',
        'num_prefix_tokens': 5,
        'source': 'torch_hub',
        'hf_id': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    },
    'dinov3_vitl16':{
        'name':'dinov3_vitl16',
        'embedding_size':1024,
        'patch_size':16,
        'hub_repo': 'facebookresearch/dinov3',
        'num_prefix_tokens': 5,
        'source': 'torch_hub',
        'hf_id': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    },
    'dinov3_vith16plus':{
        'name':'dinov3_vith16plus',
        'embedding_size':1280,
        'patch_size':16,
        'hub_repo': 'facebookresearch/dinov3',
        'num_prefix_tokens': 5,
        'source': 'torch_hub',
        'hf_id': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
    },
    'dinov3_vit7b16':{
        'name':'dinov3_vit7b16',
        'embedding_size':4096,
        'patch_size':16,
        'hub_repo': 'facebookresearch/dinov3',
        'num_prefix_tokens': 5,
        'source': 'torch_hub',
        'hf_id': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
    },
}

class conv_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(conv_head, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 64, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3,3), padding=(1,1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        x = torch.sigmoid(x)
        return x
    
class _StubDino(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)

    def get_intermediate_layers(self, x, n, reshape=True):
        b, _, h, w = x.shape
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        feat = self.proj(x)
        feat = F.interpolate(feat, size=(h_p, w_p), mode='bilinear', align_corners=False)
        return [feat for _ in n]


class Segmentor(nn.Module):
    def __init__(self, num_classes=5, backbone='dinov2_s', head='conv', backbones=DINO_BACKBONES, dino_cfg=None):
        super(Segmentor, self).__init__()
        self.heads = {
            'conv':conv_head
        }

        dino_cfg = dino_cfg or {}
        backbone_key = dino_cfg.get('backbone', backbone)
        backbone_cfg = backbones.get(backbone_key)
        if backbone_cfg is None and dino_cfg.get('source') != 'stub':
            raise ValueError(f"Unknown DINO backbone '{backbone_key}'.")

        self.layer_indices = list(dino_cfg.get('layer_indices', [8, 9, 10, 11]))
        self.num_layers = len(self.layer_indices)
        self.target_stride = int(dino_cfg.get('target_stride', 16))
        self.align_method = dino_cfg.get('align', 'interpolate')
        self.freeze = bool(dino_cfg.get('freeze', True))

        if dino_cfg.get('source') == 'stub':
            self.embed_dim = int(dino_cfg.get('embedding_size', 384))
            self.patch_size = int(dino_cfg.get('patch_size', 14))
            self.num_prefix_tokens = int(dino_cfg.get('num_prefix_tokens', 1))
            self.backbone = _StubDino(self.embed_dim, self.patch_size)
            self.backbone_source = 'stub'
        else:
            self.embed_dim = int(dino_cfg.get('embedding_size', backbone_cfg['embedding_size']))
            self.patch_size = int(dino_cfg.get('patch_size', backbone_cfg['patch_size']))
            self.num_prefix_tokens = int(dino_cfg.get('num_prefix_tokens', backbone_cfg.get('num_prefix_tokens', 1)))
            self.backbone_source = dino_cfg.get('source', backbone_cfg.get('source', 'torch_hub'))

            if self.backbone_source == 'torch_hub':
                repo = dino_cfg.get('hub_repo', backbone_cfg.get('hub_repo'))
                model_name = dino_cfg.get('hub_model', backbone_cfg.get('name'))
                hub_source = dino_cfg.get('hub_source', 'github')
                weights = dino_cfg.get('weights')
                hub_kwargs = {'source': hub_source} if hub_source else {}
                if weights:
                    hub_kwargs['weights'] = weights
                self.backbone = load(repo, model_name, **hub_kwargs)
            elif self.backbone_source == 'hf':
                model_id = dino_cfg.get('hf_model', backbone_cfg.get('hf_id'))
                if not model_id:
                    raise ValueError('hf_model is required for DINO HF source.')
                try:
                    from transformers import AutoModel
                except ImportError as exc:
                    raise ImportError('transformers is required for dino source "hf".') from exc
                token = dino_cfg.get('hf_token') or os.getenv('HF_TOKEN')
                kwargs = {'trust_remote_code': True}
                if token:
                    kwargs['token'] = token
                self.backbone = AutoModel.from_pretrained(model_id, **kwargs)
            else:
                raise ValueError(f"Unsupported dino source '{self.backbone_source}'.")

        self.backbone.eval()

        self.dino_feat_dim = self.embed_dim * self.num_layers
        self.conv3 = nn.Conv2d(self.dino_feat_dim, self.dino_feat_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.dino_feat_dim)
        self.relu = nn.ReLU(inplace=True)

        if self.align_method == 'learned':
            self.align_conv = nn.Conv2d(self.dino_feat_dim, self.dino_feat_dim, kernel_size=3, padding=1, bias=False)
        else:
            self.align_conv = None

    def _get_intermediate_layers(self, x):
        if self.backbone_source == 'hf':
            outputs = self.backbone(x, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            tokens = []
            for idx in self.layer_indices:
                if idx < 0:
                    idx = len(hidden_states) + idx
                if idx >= len(hidden_states):
                    idx = len(hidden_states) - 1
                h = hidden_states[idx]
                if h.dim() == 3:
                    h = h[:, self.num_prefix_tokens:, :]
                    b, n, c = h.shape
                    h_p = x.shape[-2] // self.patch_size
                    w_p = x.shape[-1] // self.patch_size
                    h = h.transpose(1, 2).reshape(b, c, h_p, w_p)
                tokens.append(h)
            return tokens

        return self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True)

    def _align(self, feat, target_size):
        if feat.shape[-2:] == target_size:
            return feat
        if self.align_method == 'learned' and self.align_conv is not None:
            feat = self.align_conv(feat)
        if self.align_method in ('learned', 'interpolate'):
            return F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
        raise ValueError(f"Unsupported align method '{self.align_method}'.")
                
    def forward(self, x):
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            tokens = self._get_intermediate_layers(x)
            f16 = torch.cat(tokens, dim=1)

            f16 = self.conv3(f16)
            f16 = self.bn3(f16)
            f16 = self.relu(f16)

            target_size = (x.shape[-2] // self.target_stride, x.shape[-1] // self.target_stride)
            f16 = self._align(f16, target_size)

        return f16

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class CrossChannelAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.to_q_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)

        self.to_k = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.to_k_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)

        self.to_v = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.to_v_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim*2, dim,1,1,0),
        )

    def forward(self, encoder, decoder):
        # h = self.heads
        b, c, h, w = encoder.shape

        q = self.to_q_dw(self.to_q(encoder))

        k = self.to_k_dw(self.to_k(decoder))
        v = self.to_v_dw(self.to_v(decoder))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.heads, h=h, w=w)

        return self.to_out(out)

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        
        x = swish(x)
        # x = x * torch.sigmoid(x)

        x = self.conv1(x)
        x = self.norm2(x)

        x = swish(x)
        # x = x * torch.sigmoid(x)

        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in
    
class Fuse(nn.Module):
    def __init__(self, dine_feat, out_feat):
        # need to key same channel and HW for enc / dnc
        super(Fuse, self).__init__()

        self.encode_enc = nn.Conv2d(dine_feat, out_feat, kernel_size=3, stride=1, padding=1)

        self.dim = out_feat
        self.norm1 = LayerNorm2d(self.dim)
        self.norm2 = LayerNorm2d(self.dim)

        self.dine_feat = dine_feat
        self.out_feat = out_feat
        self.crossattn = CrossChannelAttention(dim=out_feat)

        self.norm3 = LayerNorm2d(self.dim)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, enc, dnc):
        enc = self.encode_enc(enc)

        res = enc
        enc = self.norm1(enc)
        dnc = self.norm2(dnc)
        output = self.crossattn(enc, dnc) + res

        output = self.norm3(output)
        output = self.relu3(output)

        return output
