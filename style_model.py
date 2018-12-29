import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

# =========================================================================
#   Define layer
# =========================================================================
class FC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.utils.spectral_norm(nn.Linear(in_channels, out_channels))

    def forward(self, x):
        out = F.leaky_relu(self.main(x), 0.2, inplace = True)
        return out
    
class AdaIN(nn.Module):
    def forward(self, x, y_mean, y_std):
        flat_x = x.view(x.size(0), x.size(1), -1)
        x_mean = flat_x.mean(dim = -1, keepdim = True)
        x_std  = flat_x.std(dim = -1, keepdim = True)
        x_mean = x_mean.unsqueeze(-1).expand_as(x)
        x_std = x_std.unsqueeze(-1).expand_as(x)
        y_mean = y_mean.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        y_std  = y_std.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        out = (x - x_mean) / x_std * y_std + y_mean
        return out

class SynthesisLayer(nn.Module):
    def __init__(self, in_channels, out_channels, z_dims = 512, last = nn.ReLU):
        super().__init__()
        self.A_op1 = A(z_dims, in_channels)
        self.A_op2 = A(z_dims, out_channels)
        self.B_op1 = B(in_channels)
        self.B_op2 = B(out_channels)
        self.conv  = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1))
        self.AdaIN_op1 = AdaIN()
        self.AdaIN_op2 = AdaIN()
        self.last_act = last()

    def forward(self, in_tensor, w):
        out = in_tensor + self.B_op1(in_tensor)
        y_mean, y_std = self.A_op1(w)
        out = self.AdaIN_op1(out, y_mean, y_std)
        out = F.leaky_relu(self.conv(out), 0.2, inplace = True)
        out = out + self.B_op2(out)
        y_mean, y_std = self.A_op2(w)
        out = self.AdaIN_op2(out, y_mean, y_std)
        out = self.last_act(out)
        return out

# =========================================================================
#   Define module
# =========================================================================
class A(nn.Module):
    def __init__(self, z_dims, out_channels):
        super().__init__()
        self.main = nn.utils.spectral_norm(nn.Linear(z_dims, out_channels * 2))
        self.scale_head = nn.Linear(out_channels * 2, out_channels)
        self.bias_head  = nn.Linear(out_channels * 2, out_channels)

    def forward(self, w):
        hidden = F.leaky_relu(self.main(w), 0.2, inplace = True)
        scale  = self.scale_head(hidden)
        bias   = self.bias_head(hidden)
        return scale, bias

class B(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.scale_param = nn.Parameter(torch.randn([1, in_channels, 1, 1]))

    def forward(self, t):
        noise = torch.randn(t.size()).to(t.device)
        out = torch.mul(noise, self.scale_param)
        return out

# =========================================================================
#   Define sub-network
# =========================================================================
class MappingNetwork(nn.Module):
    def __init__(self, z_dims = 512):
        super().__init__()
        self.z_dims = z_dims
        self.main = nn.Sequential(
                FC(z_dims, 256),
                FC(256, 128),
                FC(128, 64),
                FC(64, 32),
                FC(32, 64),
                FC(64, 128),
                FC(128, 256),
                FC(256, 512)
        )

    def forward(self, x):
        out = self.main(x)
        return out

class SynthesisNetwork(nn.Module):
    def __init__(self, z_dims):
        super().__init__()
        self.const_input = nn.Parameter(torch.randn([128, 4, 4]))
        self.conv1 = SynthesisLayer(128, 64)                    # [ 4,  4, 128]  => [  8,   8, 64]
        self.conv2 = SynthesisLayer(64, 32)                     # [ 8,  8,  64]  => [ 16,  16, 32]
        self.conv3 = SynthesisLayer(32, 16)                     # [16, 16,  32]  => [ 32,  32, 16]
        self.conv4 = SynthesisLayer(16, 8)                      # [32, 32,  16]  => [ 64,  64,  8]
        self.conv5 = SynthesisLayer(8, 3, last = nn.Tanh)       # [64, 64,   8]  => [128, 128,  4]

    def forward(self, w):
        input = torch.stack([self.const_input] * w.size(0), 0)
        out = self.conv1(input, w)
        out = self.conv2(out, w)
        out = self.conv3(out, w)
        out = self.conv4(out, w)
        out = self.conv5(out, w)
        return out

# =========================================================================
#   Define generator
# =========================================================================
class StyleGenerator(nn.Module):
    def __init__(self, z_dims = 512):
        super().__init__()
        self.network_f = MappingNetwork(z_dims)
        self.network_g = SynthesisNetwork(z_dims)

    def forward(self, z):
        w = self.network_f(z)
        img = self.network_g(w)
        return img