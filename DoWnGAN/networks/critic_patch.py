import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    )

class Critic(nn.Module):
    """
    Shared trunk + PatchGAN heads:
      - joint head: all variables together
      - per-variable heads: encourage each variable marginal realism
      - optional global head: coarse-scale realism
    """

    def __init__(self, coarse_dim, nc_LR, nc_HR, inv_dim, use_global_head=True):
        super().__init__()
        self.coarse_dim = coarse_dim
        self.n_vars = nc_HR
        self.use_global_head = use_global_head

        # HR features (input = HR vars + HR covariates)
        self.features_hr = nn.Sequential(
            conv_block(nc_HR + inv_dim, coarse_dim, stride=1),      # 128
            conv_block(coarse_dim, coarse_dim, stride=2),           # 64
            conv_block(coarse_dim, 2*coarse_dim, stride=1),
            conv_block(2*coarse_dim, 2*coarse_dim, stride=2),       # 32
            conv_block(2*coarse_dim, 4*coarse_dim, stride=1),
            conv_block(4*coarse_dim, 4*coarse_dim, stride=2),       # 16
            conv_block(4*coarse_dim, 8*coarse_dim, stride=1),
            conv_block(8*coarse_dim, 8*coarse_dim, stride=2),       # 8
        )

        # LR features (input = LR covariates)
        # NOTE: You can tune strides here so output spatial size matches out_hr.
        self.features_lr = nn.Sequential(
            conv_block(nc_LR, coarse_dim, stride=1),
            conv_block(coarse_dim, 2*coarse_dim, stride=1),
            conv_block(2*coarse_dim, 2*coarse_dim, stride=2),       # /2
            conv_block(2*coarse_dim, 4*coarse_dim, stride=1),
        )

        # Merge and shared trunk after concat
        # out_hr has 8*cd; out_lr has 4*cd -> concat is 12*cd
        self.features_all = nn.Sequential(
            conv_block(12*coarse_dim, 12*coarse_dim, stride=1),
            conv_block(12*coarse_dim, 12*coarse_dim, stride=1),
        )

        # Patch heads (1x1 conv to score map)
        self.head_joint = nn.Conv2d(12*coarse_dim, 1, kernel_size=1)

        # Per-variable heads:
        # simplest: each head sees shared features only (cheap, effective)
        self.head_vars = nn.ModuleList([
            nn.Conv2d(12*coarse_dim, 1, kernel_size=1) for _ in range(nc_HR)
        ])

        # Optional: global head (scalar) from shared features
        if use_global_head:
            self.head_global = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(12*coarse_dim, 1, kernel_size=1),
            )

    def forward(self, x_hr_vars, cov_hr, cov_lr):
        # x_hr_vars: (B, n_vars, H, W)
        in_hr = torch.cat([x_hr_vars, cov_hr], dim=1)
        out_hr = self.features_hr(in_hr)            # (B, 8cd, 8, 8) if H=W=128

        out_lr = self.features_lr(cov_lr)
        if out_lr.shape[-2:] != out_hr.shape[-2:]:
            out_lr = F.interpolate(out_lr, size=out_hr.shape[-2:], mode="bilinear", align_corners=False)

        out = torch.cat([out_hr, out_lr], dim=1)    # (B, 12cd, 8, 8)
        feat = self.features_all(out)

        # Patch maps
        s_joint = self.head_joint(feat)             # (B, 1, 8, 8)
        s_vars = [h(feat) for h in self.head_vars]  # list of (B, 1, 8, 8)

        # Optional scalar
        s_global = self.head_global(feat).squeeze(-1).squeeze(-1) if self.use_global_head else None

        return s_joint, s_vars, s_global
