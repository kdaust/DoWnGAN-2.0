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
    # coarse_dim_n, fine_dim_n, n_covariates, n_predictands
    def __init__(
        self,
        filters,
        fine_dims,
        channels,
        channels_hr_cov=1,
        n_predictands=2,
        num_res_blocks=14,
        num_res_blocks_fine=1,
        num_upsample=3,
    ):
        super().__init__()
        self.fine_res = fine_dims
        self.n_predictands = n_predictands
        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        self.conv1f = nn.Conv2d(
            channels_hr_cov, filters, kernel_size=3, stride=1, padding=1
        )
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[
                ResidualInResidualDenseBlock(filters, noise = True, resolution=filters)
                for _ in range(num_res_blocks)
            ]
        )
        self.res_blocksf = nn.Sequential(
            *[
                ResidualInResidualDenseBlock(filters, noise = True, resolution=fine_dims)
                for _ in range(num_res_blocks_fine)
            ]
        )

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.LR_pre = nn.Sequential(
            self.conv1, ShortcutBlock(nn.Sequential(self.res_blocks, self.conv2))
        )
        self.HR_pre = nn.Sequential(
            self.conv1f, ShortcutBlock(nn.Sequential(self.res_blocksf, self.conv2))
        )

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3_all = nn.Sequential(
            nn.Conv2d(filters * 2, filters * 2, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters * 2, noise = True, resolution=fine_dims),
        )

        self.conv3_sep = nn.Sequential(
            nn.Conv2d(filters * 2, filters, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters, noise = True, resolution=fine_dims),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters, noise = False, resolution=fine_dims),
            nn.LeakyReLU(),
            nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x_coarse, x_fine):
        out = self.LR_pre(x_coarse)
        outc = self.upsampling(out)
        outf = self.HR_pre(x_fine)
        out = torch.cat((outc,outf),1)
        out = self.conv3_all(out)
        out_ls = []
        for i in range(self.n_predictands):
            out_ls.append(self.conv3_sep(out))
        res = torch.cat(out_ls, dim = 1)
        return res
