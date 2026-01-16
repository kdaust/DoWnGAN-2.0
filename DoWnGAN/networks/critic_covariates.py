# Borrowed from:
# https://github.com/Lornatang/SRGAN-PyTorch/
# Adopted to ESRGAN: https://arxiv.org/abs/1809.00219

import torch
import torch.nn as nn


class Critic(nn.Module):
    r"""The main architecture of the discriminator. Similar to VGG structure."""

    def __init__(self, coarse_dim, fine_dim, nc_LR, nc_HR, inv_dim):
        super(Critic, self).__init__()
        self.coarse_dim = coarse_dim
        self.fine_dim = fine_dim
        self.nc_HR = nc_HR
        self.nc_LR = nc_LR
        self.inv_dim = inv_dim

        self.features_hr = nn.Sequential(
            nn.Conv2d(
                self.nc_HR + inv_dim, self.coarse_dim, kernel_size=3, stride=1, padding=1
            ),  # input is 128 * 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                #int(self.coarse_dim*4),
                self.coarse_dim,
                self.coarse_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. 64
            #nn.LayerNorm([self.coarse_dim, 64, 64]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                self.coarse_dim,
                2 * self.coarse_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                2 * self.coarse_dim,
                2 * self.coarse_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. 32
            #nn.LayerNorm([2 * self.coarse_dim, 32, 32]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                2 * self.coarse_dim,
                4 * self.coarse_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                4 * self.coarse_dim,
                4 * self.coarse_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. 16
            #nn.LayerNorm([4 * self.coarse_dim, 16, 16]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                4 * self.coarse_dim,
                8 * self.coarse_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                8 * self.coarse_dim,
                8 * self.coarse_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. (512) x 6 x 6
            #nn.LayerNorm([8 * self.coarse_dim, 8, 8]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.features_lr = nn.Sequential(
            nn.Conv2d(
                self.nc_LR, self.coarse_dim, kernel_size=3, stride=1, padding=1
            ),  # input is 128 * 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                self.coarse_dim,
                2 * self.coarse_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                2 * self.coarse_dim,
                2 * self.coarse_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. 8
            #nn.LayerNorm([2 * self.coarse_dim, 32, 32]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                2 * self.coarse_dim,
                4 * self.coarse_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.features_all = nn.Sequential(
            nn.Conv2d(
                12 * self.coarse_dim,
                12 * self.coarse_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(int((self.coarse_dim*12)*(self.fine_dim/2**4)**2), 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, input: torch.Tensor, cov_hr: torch.Tensor, cov_lr: torch.Tensor) -> torch.Tensor:
        in_hr = torch.cat([input, cov_hr], 1)
        out_hr = self.features_hr(in_hr)
        out_lr = self.features_lr(cov_lr)
        out_all = torch.cat([out_hr,out_lr],1)
        out_all = self.features_all(out_all)
        out = torch.flatten(out_all, 1)
        out = self.classifier(out)

        return out
