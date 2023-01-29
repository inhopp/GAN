import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, channels=1, img_size=28):
        super(Generator, self).__init__()
        self.img_shape = (channels, img_size, img_size)

        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]

            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        # z = noise vector
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape) # (batch_size, img_shape...)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels=1, img_size=28):
        super(Discriminator, self).__init__()
        self.img_shape = (channels, img_size, img_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1) # (batch_size, flat..)
        validity = self.model(img_flat)

        return validity