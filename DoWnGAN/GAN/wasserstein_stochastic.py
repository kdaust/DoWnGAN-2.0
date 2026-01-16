from DoWnGAN.GAN.stage import StageData
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
from DoWnGAN.GAN.losses import content_loss#, kinetic_energy_loss
from DoWnGAN.mlflow_tools.gen_grid_plots import gen_grid_images
from DoWnGAN.mlflow_tools.mlflow_epoch import post_epoch_metric_mean, gen_batch_and_log_metrics, initialize_metric_dicts, log_network_models

import torch
from torch.autograd import grad as torch_grad

import mlflow
highres_in = True
torch.autograd.set_detect_anomaly(True)


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
        
        c_real = self.C(fine) ##make prediction for real image
        c_fake = self.C(fake) ##make prediction for generated image
        
        gradient_penalty = hp.gp_lambda * self._gp(fine, fake, self.C) 
            
        # Zero the gradients
        self.C_optimizer.zero_grad()
        c_real_mean = torch.mean(c_real)
        c_fake_mean = torch.mean(c_fake)

        critic_loss = c_fake_mean - c_real_mean + gradient_penalty
        critic_loss.backward(retain_graph = True)
        # Update the critic
        self.C_optimizer.step()


    def _generator_train_iteration(self, coarse, fine, invariant):
        """
        Performs one iteration of the generator training.
        Args:
            coarse (torch.Tensor): The coarse input.
            fine (torch.Tensor): The fine input.
        """
        self.G_optimizer.zero_grad()
        #print("In Generator")
        #fake_means = torch.empty([coarse.shape[0],fine.shape[1],fine.shape[2],fine.shape[3]],dtype = torch.float32, device = config.device)
        #c_result = torch.empty([coarse.shape[0]], dtype = torch.float32, device = config.device)
        fake_li = []
        for img in range(coarse.shape[0]):
            coarse_rep = coarse[img,...].unsqueeze(0).repeat(coarse.shape[0],1,1,1) ##same number as batchsize for now
            fake_stoch = self.G(coarse_rep,invariant).detach()
            fake_mean = torch.mean(fake_stoch,0) ##now just one image for each predictand
            fake_li.append(fake_mean)
            #print("fake_mean: ", len(fake_li), "Critic mean: ", len(c_li))
            del coarse_rep
            del fake_stoch
            del fake_mean
            #torch.cuda.empty_cache()

        fake_means = torch.stack(fake_li)
        
        fake = self.G(coarse, invariant)
        c_fake = self.C(fake)
        #print("Generator: ", fake_means.shape)
        cont_loss = content_loss(fake_means, fine, device=config.device) #content loss with stochastic mean
        
        # Add content loss and create objective function
        g_loss = -torch.mean(c_fake) * hp.gamma + hp.content_lambda * cont_loss

        g_loss.backward()

        # Update the generator
        self.G_optimizer.step()


    def _gp(self, real, fake, critic):
        current_batch_size = real.size(0)

        # Calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, requires_grad=True, device=config.device)
        alpha = alpha.expand_as(real)

        interpolated = alpha * real.data + (1 - alpha) * fake.data

        # Calculate probability of interpolated examples
        critic_interpolated = critic(interpolated)

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
        return hp.gp_lambda * torch.mean((gradients_norm - 1) ** 2)


    def _train_epoch(self, dataloader, testdataloader, epoch):
        """
        Performs one epoch of training.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            epoch (int): The epoch number.
        """
        print(80*"=")
        train_metrics = initialize_metric_dicts({})
        test_metrics = initialize_metric_dicts({})

        for data in dataloader:
            coarse = data[0].to(config.device)
            fine = data[1].to(config.device)
            if(highres_in):
                invariant = data[2].to(config.device)
            else:
                invariant = None
            
            self._critic_train_iteration(coarse, fine, invariant)

            if self.num_steps%hp.critic_iterations == 0:
                self._generator_train_iteration(coarse, fine, invariant)

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

        if epoch % 10 == 0:
            # Take mean of all batches and log to file
            with torch.no_grad():
                post_epoch_metric_mean(train_metrics, "train")
    
                # Generate plots from training set
                cbatch, rbatch, invbatch = next(iter(dataloader))
                if(not highres_in):
                    invbatch = -1
                gen_grid_images(self.G, cbatch, invbatch, rbatch, epoch, "train")
    
                test_metrics = initialize_metric_dicts({})
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
