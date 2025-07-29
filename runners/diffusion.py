import os
import logging
import time
import glob
import numpy as np
import torch
import sys
import cv2
import torch.nn.functional as F


sys.path.append('..')
from models.MTGD import Model
from models.load_public_dataset_all import load_data
from models.resample import LossSecondMomentResampler, LossAwareSampler, UniformSampler
from functions import get_optimizer
from functions.losses import loss_registry



def pad_tensor(tensor):
    # ?~[??| ~G维度?~X? (z, 1, x, y)
    target_stack_size = 224

    padded_tensor = F.pad(tensor, ((3,3,21,21,0,0,0,0)), mode='constant', value=0)
    return padded_tensor


# from [-1,1] to [0,1]
def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


# get beta
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]


        self.schedule_sampler = LossSecondMomentResampler(self.num_timesteps)

        # parameters
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        # variance
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # because β always lager than βbar
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config

        train_loader = load_data(config.mri.mri_sequence,config.pet.pet_modalities,config.folder_path.path)

        model = Model(self.config).to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())


        start_epoch, step = 0, 0

        # resume_training
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]


        # start training
        model.train()
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            minibatch = self.config.training.batch_size

            for i, (input_img, labels,pet_type) in enumerate(train_loader):

                input_img = input_img.squeeze(0).permute(3,0,1,2)
                input_img = F.interpolate(input_img, size=(224, 224), mode='bilinear', align_corners=False).unsqueeze(2)
                
                labels=labels.permute(3,0,1,2)
                labels = F.interpolate(labels, size=(224, 224), mode='bilinear', align_corners=False)

                index = 0

                while index< len(labels):
                    start = index
                    stop = index + minibatch
                    index += minibatch
                    if stop >= len(labels):
                        stop = len(labels)

                    mini_labels = labels[start:stop]
                    mini_inputs = input_img[start:stop]

                    batch_sums = mini_labels.sum(dim=(1, 2,3))
                    non_zero_indices = batch_sums != 0

                    # remove the slices that all black
                    if torch.sum(non_zero_indices == False).item() == (stop - start):
                        continue
                    else:
                        mini_labels = mini_labels[non_zero_indices]
                        mini_inputs = mini_inputs[non_zero_indices]

                    # last slice
                    if len(mini_inputs.shape)==3:
                        mini_labels = mini_labels.unsqueeze(0)
                        mini_inputs = mini_inputs.unsqueeze(0)
                    n = mini_labels.size(0)
                    data_time += time.time() - data_start

                    step += 1
                    mini_labels = mini_labels.to(self.device).float()
                    mini_inputs = mini_inputs.to(self.device).float()
                    pet_type_batch = pet_type.to(self.device).float().repeat_interleave(n)

                    e = torch.randn_like(mini_labels)
                    b = self.betas

                    t, weights = self.schedule_sampler.sample(mini_labels.shape[0], self.device)

                    # model,x0,t,noise,beta
                    loss = loss_registry[config.model.type](model, mini_labels, t,e, b, mini_inputs, pet_type_batch,keepdim=True, loss_moment=True)

                    if isinstance(self.schedule_sampler, LossAwareSampler):
                        self.schedule_sampler.update_with_local_losses(
                            t, loss.detach()
                        )

                    w_loss = (loss * weights).mean(dim=0)

                    logging.info(
                        f"step: {step}, loss: {loss.mean(dim=0):.6f}, weight loss: {w_loss:.6f},data time: {data_time / (i + 1):.6f}"
                    )

                    optimizer.zero_grad()
                    w_loss.backward()

                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()

                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]

                        torch.save(
                            states,
                            os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                        )
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                    data_start = time.time()



    def sample(self):
        args, config = self.args, self.config
        model = Model(self.config).to(self.device)


        try:
            states = torch.load(
                os.path.join(self.args.log_path, "ckpt.pth"),
                map_location=self.config.device,
            )
            model.load_state_dict(states[0], strict=True)

        except:
            logging.info(
                'could not load parameters'
            )
            exit()
        model = model.to(self.device)


        logging.info('loading parameters successfully')
        model.eval()
        test_loader = load_data(config.mri.mri_sequence, config.pet.pet_modalities, config.folder_path.path,False)

        self.sample_sequence(model, test_loader)

    def sample_sequence(self, model, test_loader):
        minibatch = self.config.sampling.batch_size

        logging.info('start generating images')
        for i, (condition, x0,pet_type) in enumerate(test_loader):
            condition = condition.squeeze(0).permute(3, 0, 1, 2)
            condition = F.interpolate(condition, size=(224, 224), mode='bilinear', align_corners=False).unsqueeze(2)

            x0 = x0.permute(3, 0, 1, 2)
            x0 = F.interpolate(x0, size=(224, 224), mode='bilinear', align_corners=False)

            index = 0
            while index < len(x0):

                start = index
                stop = index + minibatch
                index = stop
                if stop >= len(x0):
                    stop = len(x0)

                x0_mini = x0[start:stop]
                condition_mini = condition[start:stop]

                batch_sums = x0_mini.sum(dim=(1, 2,3))
                non_zero_indices = batch_sums != 0

            
                if torch.sum(non_zero_indices == False).item() == (stop - start):
                    continue
                else:
                    condition_mini = condition_mini[non_zero_indices]
                    x0_mini = x0_mini[non_zero_indices]

                if len(condition_mini.shape) == 3:
                    x0_mini = x0_mini.unsqueeze(0)
                    condition_mini = condition_mini.unsqueeze(0)

                x0_mini, condition_mini = x0_mini.to(self.device).float(), condition_mini.to(self.device).float()

                # random noise, star noise
                x = torch.randn(
                    (len(x0_mini)),  # batch_size
                    1,  # data.channels
                    x0_mini.shape[2],  # image_size
                    x0_mini.shape[3],  # image_size
                    device=self.device,
                )


                with torch.no_grad():
 
                    print(condition_mini.shape,x.shape)
                    pet_type_batch = pet_type.to(self.device).float().repeat_interleave(x.shape[0])
                    
                    x = self.sample_image(x, model, condition_mini, pet_type_batch,last=True)

                    x = (x * 255.0).clamp(0, 255).to(torch.uint8)
                    x0_mini = (x0_mini * 255.0).clamp(0, 255).to(torch.uint8)
                    
                    x = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    x0_mini = x0_mini.permute(0, 2, 3, 1).contiguous().cpu().numpy()

                    images = np.concatenate([x, x0_mini], axis=2)
                    index = self.save_img(images,i,index)


    def sample_image(self, x, model, condition, pet_type_batch,last=True):

        skip = 1
        seq = range(0, self.num_timesteps, skip)

        from functions.denoising import ddpm_steps

        _, x = ddpm_steps(x, seq, model, self.betas, condition,pet_type_batch)
        return x[-1]

    def save_img(self,imgs,p_idx,idx):
    
        folder=os.path.join(self.args.image_folder, str(p_idx))
        
        if not os.path.exists(folder):
                os.makedirs(folder)
                
        for mini_index in range(len(imgs)):
            cv2.imwrite(os.path.join(folder, str(idx)) + '.png', imgs[mini_index])
            idx+=1
            
        return idx
    
