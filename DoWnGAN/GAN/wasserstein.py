from DoWnGAN.GAN.stage import StageData
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
from DoWnGAN.GAN.losses import content_loss, crps_empirical
from DoWnGAN.mlflow_tools.gen_grid_plots import gen_grid_images
from DoWnGAN.mlflow_tools.mlflow_epoch import post_epoch_metric_mean, gen_batch_and_log_metrics, initialize_metric_dicts, log_network_models
import torch
from torch.autograd import grad as torch_grad

import mlflow
highres_in = True
freq_sep = False
torch.autograd.set_detect_anomaly(True)
n_realisation = 4 ##stochastic sampling

class WassersteinGAN:
    """Implements Wasserstein GAN with gradient penalty and 
    frequency separation"""

    def __init__(self, G, C, G_optimizer, C_optimizer) -> None:
        self.G = G
        self.C = C
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
        self.num_steps = 0
        
    def _critic_train_iteration(self, coarse, fine, invariant):
        """
        Performs one iteration of the critic training.
        Args:
            coarse (torch.Tensor): The coarse input.
            fine (torch.Tensor): The fine input.
        """
        if(highres_in):
            fake = self.G(coarse, invariant) ##generate fake image from generator
        else:
            fake = self.G(coarse)
        c_real = self.C(fine,invariant,coarse) ##make prediction for real image
        c_fake = self.C(fake,invariant,coarse) ##make prediction for generated image

        gradient_penalty = hp.gp_lambda * self._gp(fine, fake, self.C, coarse, invariant) 
            
        # Zero the gradients
        self.C_optimizer.zero_grad()
        c_real_mean = torch.mean(c_real)
        c_fake_mean = torch.mean(c_fake)

        critic_loss = c_fake_mean - c_real_mean + gradient_penalty
        critic_loss.backward(retain_graph = True)
        # Update the critic
        self.C_optimizer.step()


    def _generator_train_iteration(self, coarse, fine, invariant, iteration):
        """
        Performs one iteration of the generator training.
        Args:
            coarse (torch.Tensor): The coarse input.
            fine (torch.Tensor): The fine input.
        """
        self.G_optimizer.zero_grad()
        
        if(highres_in):
            fake = self.G(coarse, invariant) ##generate fake image from generator
        else:
            fake = self.G(coarse)
        
        if(freq_sep):
            fake_low = hp.low(hp.rf(fake))
            real_low = hp.low(hp.rf(fine))

            c_fake = self.C(fake,invariant,coarse)
            cont_loss = content_loss(fake_low, real_low, device=config.device)
            #cont_loss = content_loss(fake, fine, device=config.device)
            # ls1 = [i for i in range(fine.shape[0])]
            # dat_lr = [coarse[i,...].unsqueeze(0).repeat(n_realisation,1,1,1) for i in ls1]
            # dat_hr = [fine[i,...] for i in ls1]
            # dat_sr = [self.G(lr,invariant[0:n_realisation,...]) for lr in dat_lr]
            # crps_ls = [crps_empirical(sr,hr) for sr,hr in zip(dat_sr,dat_hr)]
        else: ##stochastic mean
            c_fake = self.C(fake,invariant,coarse) ## wasserstein distance
            dat_gen = [self.G(coarse, invariant) for i in range(n_realisation)]
            dat_sr = [torch.cat([x[ind,...].unsqueeze(0) for x in dat_gen],dim = 0) for ind in range(coarse.shape[0])]
            dat_hr = [fine[i,...] for i in range(coarse.shape[0])]
            crps_ls = [crps_empirical(sr,hr) for sr,hr in zip(dat_sr,dat_hr)]
            crps = torch.cat(crps_ls)
        
        #g_loss = -torch.mean(c_fake) * hp.gamma + hp.content_lambda * cont_loss
        g_loss = -torch.mean(c_fake) * hp.gamma + hp.content_lambda * torch.mean(crps)
        
        g_loss.backward()

        # Update the generator
        self.G_optimizer.step()



    def _gp(self, real, fake, critic, coarse, invariant):
        current_batch_size = real.size(0)

        # Calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, requires_grad=True, device=config.device)
        alpha = alpha.expand_as(real)

        interpolated = alpha * real.data + (1 - alpha) * fake.data

        # Calculate probability of interpolated examples
        critic_interpolated = critic(interpolated, invariant, coarse)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), device=config.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(hp.batch_size, -1).to(config.device)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return torch.mean((gradients_norm - 1) ** 2)


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
            if(highres_in):
                invariant = data[2].to(config.device)
            else:
                invariant = None
            
            self._critic_train_iteration(coarse, fine, invariant)

            if self.num_steps%hp.critic_iterations == 0:
                self._generator_train_iteration(coarse, fine, invariant, epoch)

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
                if(not highres_in):
                    invbatch = -1
                gen_grid_images(self.G, cbatch, invbatch, rbatch, epoch, "train")
    
                test_metrics = initialize_metric_dicts({}, rbatch.shape[1])
                for data in testdataloader:
                    coarse = data[0].to(config.device)
                    fine = data[1].to(config.device)
                    if(highres_in):
                        invariant = data[2].to(config.device)
                    else:
                        invariant = None
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
                if(not highres_in):
                    invbatch = -1
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
