import argparse
import os
import numpy as np
import math
import sys
from datetime import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='', help='path to training data')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class MyDataset(Dataset):
    def __init__(self, data_path:str, split:str, transform: Callable, **kwargs):
        self.data_dir = Path(data_path)
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.png'])
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, 0.0


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
if opt.data:
    if opt.channels == 1:
        my_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    else:
        my_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    mydataset = MyDataset(
        data_path=opt.data,
        split='train',
        transform=my_transform,
    )
    print(len(mydataset))
    dataloader = DataLoader(mydataset,
                            batch_size=opt.batch_size,
                            shuffle=True)
else:
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
os.makedirs('saved_models', exist_ok=True)
start_time = time.time()
batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            batches_done += opt.n_critic
            estimated_time_per_batch = (time.time() - start_time) / batches_done
            remain_batches = len(dataloader) - (i+1) + (opt.n_epochs - epoch - 1)*len(dataloader)
            remain_time = remain_batches * estimated_time_per_batch
            remain_hr = remain_time // 3600
            remain_min = (remain_time % 3600) // 60
            remain_sec = (remain_time % 3600) % 60
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Remaining time: %d:%d:%d]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), remain_hr, remain_min, remain_sec)
            )
            sys.stdout.flush()

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                # Save to checkpoint
                checkpoint_path = 'saved_models/checkpoint.pt'
                torch.save({
                    'epoch': opt.n_epochs,
                    'G_model_state_dict': generator.state_dict(),
                    'G_optimizer_state_dict': optimizer_G.state_dict(),
                    'G_loss': loss_G.item(),
                    'D_model_state_dict': discriminator.state_dict(),
                    'D_optimizer_state_dict': optimizer_D.state_dict(),
                    'D_loss': loss_D.item(),
                }, checkpoint_path)

# Save the model
now = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
model_path = 'saved_models/generator_{}.pth'.format(now)
torch.save(generator.state_dict(), model_path)
