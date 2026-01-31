from DoWnGAN.GAN.stage import StageData
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
from DoWnGAN.GAN.losses import content_loss, crps_empirical
from DoWnGAN.mlflow_tools.gen_grid_plots import gen_grid_images
from DoWnGAN.mlflow_tools.mlflow_epoch import post_epoch_metric_mean, gen_batch_and_log_metrics, initialize_metric_dicts, log_network_models, CriticGapCSVLogger, compute_critic_gaps
import torch
from torch.autograd import grad as torch_grad

import mlflow
highres_in = True
freq_sep = False
torch.autograd.set_detect_anomaly(True)
n_realisation = 4 ##stochastic sampling


class WassersteinGAN:
    """Implements Wasserstein PatchGAN with gradient penalty and 
    stochastic CRPS loss"""

    def __init__(self, G, C, G_optimizer, C_optimizer) -> None:
        self.G = G
        self.C = C
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
        self.num_steps = 0
        self.critic_logger = None

    def set_critic_gap_logger(self, logger):
        self.critic_logger = logger
        
    def critic_score(self, s_joint, s_vars, s_global):
        """
        Reduce multi-head PatchGAN outputs to scalar D(x) per sample.

        Returns:
            D: (B,) tensor
        """
        # Joint patch -> scalar per sample
        D = hp.lambdas["joint"] * s_joint.mean(dim=(1, 2, 3))  # (B,)

        # Per-variable patches -> scalar per sample
        for i, s in enumerate(s_vars):
            D = D + hp.lambdas["vars"][i] * s.mean(dim=(1, 2, 3))

        # Optional global scalar
        if s_global is not None:
            D = D + hp.lambdas.get("global", 0.0) * s_global.view(-1)

        return D
    
    def _critic_train_iteration(self, coarse, fine, invariant, iteration):
        """
        coarse: LR covariates (your cov_lr)
        fine:   HR target vars (x_hr_vars)
        invariant: HR covariates (your cov_hr)
        """
        self.C_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            fake = self.G(coarse, invariant)

        s_joint_real, s_vars_real, s_global_real = self.C(fine, invariant, coarse)
        s_joint_fake, s_vars_fake, s_global_fake = self.C(fake, invariant, coarse)

        D_real = self.critic_score(s_joint_real, s_vars_real, s_global_real)
        D_fake = self.critic_score(s_joint_fake, s_vars_fake, s_global_fake)
        drift = (D_real ** 2).mean() * hp.drift_epsilon ##prevent adversarial loss from getting too big
        gp = hp.gp_lambda * self._gp(fine, fake, invariant, coarse)

        if iteration % 500 == 0:
            gaps = compute_critic_gaps(
                s_joint_real, s_joint_fake,
                s_vars_real, s_vars_fake,
                s_global_real, s_global_fake,
                var_names=["u","v","T","q","P"]
            )
            self.critic_logger.log(self.num_steps, gaps)
            
        critic_loss = (D_fake.mean() - D_real.mean()) + gp + drift

        critic_loss.backward()
        self.C_optimizer.step()
    
    def _generator_train_iteration(self, coarse, fine, invariant, iteration):
        self.G_optimizer.zero_grad(set_to_none=True)

        fake = self.G(coarse, invariant)

        s_joint_fake, s_vars_fake, s_global_fake = self.C(fake, invariant, coarse)
        D_fake = self.critic_score(s_joint_fake, s_vars_fake, s_global_fake)

        if freq_sep:
            fake_low = hp.low(hp.rf(fake))
            real_low = hp.low(hp.rf(fine))
            crps_term = content_loss(fake_low, real_low, device=config.device)
        else:
            B = coarse.size(0)
            R = n_realisation

            # Repeat inputs along batch dimension
            coarse_rep = coarse.repeat_interleave(R, dim=0)      # (B*R, C_lr, H_lr, W_lr)
            inv_rep    = invariant.repeat_interleave(R, dim=0)   # (B*R, C_hr, H_hr, W_hr)

            # One forward pass for all realizations
            sr_rep = self.G(coarse_rep, inv_rep)                 # (B*R, C, H, W)

            # Reshape to (B, R, C, H, W)
            dat_gen = sr_rep.view(
                B, R, sr_rep.size(1), sr_rep.size(2), sr_rep.size(3)
            )
            dat_sr = [dat_gen[b] for b in range(B)]
            dat_hr = [fine[b] for b in range(B)]
            crps_ls = [crps_empirical(sr, hr) for sr, hr in zip(dat_sr, dat_hr)]
            crps = torch.cat(crps_ls)
            crps_term = crps.mean()

        if iteration % 500 == 0:
            print(f"Adversarial loss: {float(-D_fake.mean())}; CRPS Loss: {float(hp.content_lambda * crps_term)}")
        g_loss = -D_fake.mean() + crps_term * hp.content_lambda

        g_loss.backward()
        self.G_optimizer.step()

    def _gp(self, real, fake, cov_hr, cov_lr):
        """
        WGAN-GP gradient penalty on interpolated HR fields.
        Conditioning (cov_hr/cov_lr) is treated as constant.
        """
        B = real.size(0)
        device = real.device

        alpha = torch.rand(B, 1, 1, 1, device=device)
        x_hat = alpha * real + (1.0 - alpha) * fake
        x_hat.requires_grad_(True)

        s_joint, s_vars, s_global = self.C(x_hat, cov_hr, cov_lr)
        D_hat = self.critic_score(s_joint, s_vars, s_global)  # (B,)

        grads = torch.autograd.grad(
            outputs=D_hat.sum(),
            inputs=x_hat,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]  # (B, C, H, W)

        grads = grads.view(B, -1)
        grad_norm = grads.norm(2, dim=1)
        gp = ((grad_norm - 1.0) ** 2).mean()

        return gp

    def log_noise_strengths(self, step):
        strengths = []
        for m in self.G.modules():
            if hasattr(m, "noise_strength"):
                strengths.append(float(m.noise_strength.detach().cpu()))
        if strengths:
            print(f"[Step {step}] noise_strength mean={sum(strengths)/len(strengths):.4f}, "
                f"min={min(strengths):.4f}, max={max(strengths):.4f}")

    def _train_epoch(self, dataloader, testdataloader, epoch):
        """
        Performs one epoch of training.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            epoch (int): The epoch number.
        """
        print(80*"=")
        ##print("Wasserstein GAN")
        train_metrics = initialize_metric_dicts({},5)
        test_metrics = initialize_metric_dicts({},5)

        for i,data in enumerate(dataloader):
            coarse = data[0].to(config.device)
            fine = data[1].to(config.device)
            invariant = data[2].to(config.device)

            self._critic_train_iteration(coarse, fine, invariant, self.num_steps)

            if self.num_steps%hp.critic_iterations == 0:
                self._generator_train_iteration(coarse, fine, invariant, self.num_steps)

            # Track train set metrics
            train_metrics = gen_batch_and_log_metrics(
                self.G,
                self.C,
                coarse,
                fine,
                invariant,
                train_metrics,
            )
            self.num_steps += 1

        if epoch % 5 == 0:
            # Take mean of all batches and log to file
            with torch.no_grad():
                post_epoch_metric_mean(train_metrics, "train")
    
                # Generate plots from training set
                cbatch, rbatch, invbatch = next(iter(dataloader))
                gen_grid_images(self.G, cbatch, invbatch, rbatch, epoch, "train")
    
                test_metrics = initialize_metric_dicts({}, rbatch.shape[1])
                for data in testdataloader:
                    coarse = data[0].to(config.device)
                    fine = data[1].to(config.device)
                    invariant = data[2].to(config.device)

                    # Track train set metrics
                    test_metrics = gen_batch_and_log_metrics(
                        self.G,
                        self.C,
                        coarse,
                        fine,
                        invariant,
                        test_metrics,
                    )
    
                # Take mean of all batches and log to file
                post_epoch_metric_mean(test_metrics, "test")
    
                cbatch, rbatch, invbatch = next(iter(testdataloader))

                gen_grid_images(self.G, cbatch, invbatch, rbatch, epoch, "test")
    
                # Log the models to mlflow pytorch models
                print(f"Artifact URI: {mlflow.get_artifact_uri()}")
                log_network_models(self.C, self.G, epoch)

    def train(self, dataloader, testdataloader):
        """
        Trains the model.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
        """
        self.num_steps = 0
        for epoch in range(hp.epochs):
            self._train_epoch(dataloader, testdataloader, epoch)
