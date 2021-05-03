# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
batchSize = 64  # We set the size of the batch.
imageSize = 64  # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Loading the dataset
dataset = dset.CIFAR10(root='./data', download=True,
                       transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=2)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


generator = G()
generator.apply(weights_init)


class D(nn.Module):
    def __init(self):
        super(D, self).__init()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)  # Flattening


discriminator = D()
discriminator.apply(weights_init)


#  Training
criterion = nn.BCELoss()
d_optim = optim.Adam(discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
g_optim = optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))

for epoch in range(25):
    for x, minibatch in enumerate(dataloader, 0):
        discriminator.zero_grad()
        # Тренируем дискриминатор на настоящем изображении из датасета. Из пары элементов изображение-класс, выбираем изображение.
        real, _ = minibatch
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = discriminator(input)
        real_discriminator_error = criterion(output, target)

        # Тренируем дискриминатор на фейковых изображениях
        noise = Variable(torch.randn(input.size()[0]), 100, 1, 1)
        fake = generator(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = discriminator(fake.detach())
        fake_discriminator_error = criterion(output, target)

        # Обратное распространение ошибки
        total_discriminator_error = real_discriminator_error + fake_discriminator_error
        total_discriminator_error.backward()
        d_optim.step()

        # Обновление весов для генератора
        generator.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = discriminator(fake)
        generator_error = criterion(output, target)
        generator_error.backward()
        g_optim.step()

        print(f'[{epoch}/25][{x}/{len(dataloader)}] Loss_D: {total_discriminator_error} Loss_G: {generator_error}')




