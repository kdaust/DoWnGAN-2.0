#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaption of Nic's generator based on ESRGAN+ model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class DenseResidualBlockNoise(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, resolution, res_scale=0.8, noise_sd=1):
        super().__init__()
        self.res_scale = res_scale
        self.resolution = resolution
        self.noise_sd = noise_sd

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters + 1)
        self.b2 = block(in_features=2 * filters + 2)
        self.b3 = block(in_features=3 * filters + 3)
        self.b4 = block(in_features=4 * filters + 4)
        self.b5 = block(in_features=5 * filters + 5, non_linearity=False)
        #self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        self.noise_strength = torch.nn.Parameter(torch.mul(torch.ones([]), 10))

    def forward(self, x):
        nrm_mean = torch.zeros([x.shape[0], 1, self.resolution, self.resolution], device = x.device)
        nrm_std = torch.full([x.shape[0], 1, self.resolution, self.resolution],self.noise_sd, device = x.device)
        noise = torch.normal(
            nrm_mean,
            nrm_std,
        )
        inputs = torch.cat([x, noise], 1)

        out = self.b1(inputs)
        noise = torch.normal(
                nrm_mean,
                nrm_std,
            )
        inputs = torch.cat([inputs, out, noise], 1)
        out = self.b2(inputs)
        noise = torch.normal(
                nrm_mean,
                nrm_std,
            )
        inputs = torch.cat([inputs, out, noise], 1)
        out = self.b3(inputs)
        noise = torch.normal(
                nrm_mean,
                nrm_std,
            )
        inputs = torch.cat([inputs, out, noise], 1)
        out = self.b4(inputs)
        noise = torch.normal(
                nrm_mean,
                nrm_std,
            )
        inputs = torch.cat([inputs, out, noise], 1)
        out = self.b5(inputs)
        noise = torch.normal(
                nrm_mean,
                nrm_std,
            )
        inputs = torch.cat([inputs, out, noise], 1)

        noise = torch.normal(
            nrm_mean,
            nrm_std,
        )
        noiseScale = noise * self.noise_strength
        out = out.mul(self.res_scale) + x
        out.add_(noiseScale)
        return out


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x

        # list blocks manually for torch script
        out = self.b1(inputs)
        inputs = torch.cat([inputs, out], 1)
        out = self.b2(inputs)
        inputs = torch.cat([inputs, out], 1)
        out = self.b3(inputs)
        inputs = torch.cat([inputs, out], 1)
        out = self.b4(inputs)
        inputs = torch.cat([inputs, out], 1)
        out = self.b5(inputs)
        inputs = torch.cat([inputs, out], 1)

        # for block in self.blocks:
        #     out = block(inputs)
        #     inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, noise, resolution, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        if noise:
            self.dense_blocks = nn.Sequential(
                DenseResidualBlockNoise(filters, resolution),
                DenseResidualBlockNoise(filters, resolution),
                DenseResidualBlockNoise(filters, resolution),
            )
        else:
            self.dense_blocks = nn.Sequential(
                DenseResidualBlock(filters),
                DenseResidualBlock(filters),
                DenseResidualBlock(filters),
            )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class Generator(nn.Module):
    """
    Grouped-head variant for predictands ordered as: [u, v, T, q]

    - Shared trunk up to `conv3_all`
    - Two head trunks:
        * head_uv_shared: shared for u and v
        * head_Tq_shared: shared for T and q
    - Separate output conv per variable:
        * out_u, out_v, out_T, out_q
    """

    def __init__(
        self,
        filters,
        fine_dims,
        channels,
        channels_hr_cov=1,
        n_predictands=4,          # expects 4 here: u,v,T,q
        num_res_blocks=14,
        num_res_blocks_fine=1,
        num_upsample=3,
        noise_in_heads=True,      # you can set False if heads get too "noisy"
    ):
        super().__init__()
        assert n_predictands == 4, "This grouped-head variant assumes 4 predictands: [u, v, T, q]"
        self.fine_res = fine_dims
        self.n_predictands = n_predictands

        # First layers
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        self.conv1f = nn.Conv2d(channels_hr_cov, filters, kernel_size=3, stride=1, padding=1)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[
                ResidualInResidualDenseBlock(filters, noise=True, resolution=filters)
                for _ in range(num_res_blocks)
            ]
        )
        self.res_blocksf = nn.Sequential(
            *[
                ResidualInResidualDenseBlock(filters, noise=True, resolution=fine_dims)
                for _ in range(num_res_blocks_fine)
            ]
        )

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.LR_pre = nn.Sequential(
            self.conv1,
            ShortcutBlock(nn.Sequential(self.res_blocks, self.conv2)),
        )
        self.HR_pre = nn.Sequential(
            self.conv1f,
            ShortcutBlock(nn.Sequential(self.res_blocksf, self.conv2)),
        )

        # Upsampling layers (pixelshuffle)
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # Joint fusion at HR
        self.conv3_all = nn.Sequential(
            nn.Conv2d(filters * 2, filters * 2, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters * 2, noise=True, resolution=fine_dims),
        )

        # --- Grouped heads ---
        # Shared trunk for u/v
        self.head_uv_shared = nn.Sequential(
            nn.Conv2d(filters * 2, filters, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters, noise=noise_in_heads, resolution=fine_dims),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters, noise=False, resolution=fine_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_u = nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1)
        self.out_v = nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1)

        # Shared trunk for T/q
        self.head_Tq_shared = nn.Sequential(
            nn.Conv2d(filters * 2, filters, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters, noise=noise_in_heads, resolution=fine_dims),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters, noise=False, resolution=fine_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_T = nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1)
        self.out_q = nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_coarse, x_fine):
        # Shared trunk
        out = self.LR_pre(x_coarse)
        outc = self.upsampling(out)     # HR features from LR covariates
        outf = self.HR_pre(x_fine)      # HR covariate branch (must be same spatial res as outc)
        feat = self.conv3_all(torch.cat((outc, outf), dim=1))   # (B, 2*filters, H, W)

        # Group heads
        feat_uv = self.head_uv_shared(feat)
        u = self.out_u(feat_uv)
        v = self.out_v(feat_uv)

        feat_Tq = self.head_Tq_shared(feat)
        T = self.out_T(feat_Tq)
        q = self.out_q(feat_Tq)

        # Output order: [u, v, T, q]
        return torch.cat([u, v, T, q], dim=1)
