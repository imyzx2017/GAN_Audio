from DCGAN_Audio.model import SpecGenerator, SpecDiscriminator
import torch
import torch.nn as nn
from DCGAN_Audio.util import data_loader
import numpy as np
from torchvision.utils import save_image
# using GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 4
noise_dim = 200
G = SpecGenerator(noise_dim=noise_dim).to(DEVICE)
D = SpecDiscriminator().to(DEVICE)


criterion = nn.BCELoss()
G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# set other parameters
max_epoch = 100
step = 0
n_critic = 1    # for traing more k steps about Discriminator

# making labels
D_real_labels = torch.ones([batch_size, 1]).to(DEVICE)
D_fake_labels = torch.zeros([batch_size, 1]).to(DEVICE)


def get_sample_image(G, n_noise):
    """
        save sample 9 images
    """
    z = torch.randn(9, n_noise).to(DEVICE)
    y_hat = G(z).view(9, 512, 512) # (100, 28, 28)
    result = y_hat.cpu().data.numpy()
    img = np.zeros([512*3, 512*3])
    for j in range(3):
        img[j*512:(j+1)*512] = np.concatenate([x for x in result[j*3:(j+1)*3]], axis=-1)
    return img

def get_onesample_image(G, n_noise):
    noise = torch.rand(1, n_noise).to(DEVICE)
    fake_img = G(noise)
    return fake_img

def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)

for epoch in range(max_epoch):
    for idx, (images, labels) in enumerate(data_loader):
        # print(images)
        # training Discriminator
        x = images.to(DEVICE)
        x_outpus = D(x)
        D_x_loss = criterion(x_outpus, D_real_labels)

        z = torch.randn(batch_size, noise_dim).to(DEVICE)
        z_outputs = D(G(z))
        D_z_loss = criterion(z_outputs, D_fake_labels)   ####  here using fake labels to strength the Discriminator

        D_loss = D_x_loss + D_z_loss

        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        if step % n_critic == 0:
            # traing Generator
            z = torch.randn(batch_size, noise_dim).to(DEVICE)
            z_outputs = D(G(z))
            G_loss = criterion(z_outputs, D_real_labels)         #### here using real labels, try to generate real image

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_opt.step()

        if step % 500 == 0:
            print('Epoch: {}/{}, step: {}, D_Loss: {}, G_loss: {}'.format(epoch, max_epoch, step, D_loss.item(), G_loss.item()))

        if step % 1000 == 0:
            save_checkpoint({'epoch': epoch + 1, 'state_dict': D.state_dict(), 'optimizer': D_opt.state_dict()},
                            'D.{}_pth.tar'.format(step))
            save_checkpoint({'epoch': epoch + 1, 'state_dict': G.state_dict(), 'optimizer': G_opt.state_dict()},
                            'G.{}_pth.tar'.format(step))
            G.eval()
            # generate_img = get_sample_image(G, noise_dim)
            generate_img = get_onesample_image(G, noise_dim)
            # generate_img = generate_img.view(3, 1024, 1024)
            # imsave('./result/{}_step{}.jpg'.format('ResBlock_DCGAN', str(step)), generate_img.cpu().data.numpy())
            save_image(generate_img, 'D:\\Projects\\Projects\\pytorch_Projects\\GAN\\GAN_YZX\\DCGAN_Audio\\result_512\\{}_step{}.jpg'.format('ResBlock_DCGAN', str(step)))
            G.train()
        step += 1


