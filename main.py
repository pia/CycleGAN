import torchvision as tv
import torch as t
from model import *
from config import *
from time import time
from utils import *
import torch.nn.functional as F
import threading
import itertools
import torch
from torch.utils.data import DataLoader
from datasets import ImageDataset
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

# t.backends.cudnn.benchmark = True
start = time()
opt = Config()


def train(**kwargs):
    device = opt.device

    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.Tensor
    input_A = Tensor(opt.batch_size, 3, opt.image_size, opt.image_size)
    input_B = Tensor(opt.batch_size, 3, opt.image_size, opt.image_size)

    # data
    # transforms = tv.transforms.Compose([
    #     # tv.transforms.Resize((opt.image_size, opt.image_size)),
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transforms_ = [transforms.Resize(int(opt.image_size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.image_size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader = DataLoader(ImageDataset('datasets/horse2zebra/', transforms_=transforms_, unaligned=True),
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True)

    # network
    netg_a2b, netd_a = Generator(), Discriminator()
    netg_b2a, netd_b = Generator(), Discriminator()
    netg_a2b.initialize_weights()
    netd_a.initialize_weights()
    netg_b2a.initialize_weights()
    netd_b.initialize_weights()

    netg_a2b.to(device)
    netd_a.to(device)
    netg_b2a.to(device)
    netd_b.to(device)

    optimizer_g = t.optim.Adam(itertools.chain(netg_a2b.parameters(), netg_b2a.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d_a = t.optim.Adam(netd_a.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d_b = t.optim.Adam(netd_b.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=LambdaLR(opt.n_epochs, 0,
                                                                                       100).step)
    lr_scheduler_D_a = torch.optim.lr_scheduler.LambdaLR(optimizer_d_a, lr_lambda=LambdaLR(opt.n_epochs, 0,
                                                                                       100).step)
    lr_scheduler_D_b = torch.optim.lr_scheduler.LambdaLR(optimizer_d_b, lr_lambda=LambdaLR(opt.n_epochs, 0,
                                                                                           100).step)

    criterion_identity = t.nn.L1Loss().to(device)
    criterion_cycle = t.nn.L1Loss().to(device)
    criterion_GAN = t.nn.MSELoss().to(device)

    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)

    print('used {}s to init.'.format(time() - start))

    global_i = 0
    visual = Visualize()
    epoch_start = 0
    time_init = time()

    # load model
    if not opt.debug and len(os.listdir(opt.checkpoint_path)) > 0:
        path = find_new_file(opt.checkpoint_path)
        checkpoint = t.load(path)
        netg_a2b.load_state_dict(checkpoint['g_a2b_state_dict'])
        netg_b2a.load_state_dict(checkpoint['g_b2a_state_dict'])
        netd_a.load_state_dict(checkpoint['d_a_state_dict'])
        netd_b.load_state_dict(checkpoint['d_b_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d_a.load_state_dict(checkpoint['optimizer_d_a_state_dict'])
        optimizer_d_b.load_state_dict(checkpoint['optimizer_d_b_state_dict'])
        epoch_start = checkpoint['epoch']

    for epoch in range(epoch_start, opt.max_epoch):
        for ii, batch in enumerate(dataloader):
            # Set model input
            real_img = Variable(input_A.copy_(batch['A']))
            style_img = Variable(input_B.copy_(batch['B']))

            if (global_i % opt.g_every == 0):
                # train generator
                optimizer_g.zero_grad()

                # Identity loss
                same_b = netg_a2b(style_img)
                loss_identity_b = criterion_identity(same_b, style_img)*5.0
                same_a = netg_b2a(real_img)
                loss_identity_a = criterion_identity(same_a, real_img)*5.0

                # GAN Loss
                fake_b = netg_a2b(real_img)
                pred_fake = netd_b(fake_b)
                loss_GAN_a2b = criterion_GAN(pred_fake, true_labels)

                fake_a = netg_b2a(style_img)
                pred_fake = netd_a(fake_a)
                loss_GAN_b2a = criterion_GAN(pred_fake, fake_labels)

                # Cycle loss
                recovered_a = netg_b2a(fake_b)
                loss_cycle_aba = criterion_cycle(recovered_a, real_img)*10

                recovered_b = netg_a2b(fake_a)
                loss_cycle_bab = criterion_cycle(recovered_b, style_img)*10

                V_g = loss_identity_a + loss_identity_b + loss_GAN_a2b + loss_GAN_b2a + loss_cycle_aba + loss_cycle_bab
                V_g.backward()

                optimizer_g.step()

            if (global_i % opt.d_every == 0):
                # train discriminator
                optimizer_d_a.zero_grad()

                # Real loss
                pred_real = netd_a(real_img)
                loss_d_real = criterion_GAN(pred_real, true_labels)

                # Fake loss
                pred_fake = netd_a(fake_a.detach())
                loss_d_fake = criterion_GAN(pred_fake, fake_labels)

                V_d_a = (loss_d_real + loss_d_fake)*0.5
                V_d_a.backward()

                optimizer_d_a.step()

                optimizer_d_b.zero_grad()

                # Real loss
                pred_real = netd_b(style_img)
                loss_d_real = criterion_GAN(pred_real, true_labels)

                # Fake loss
                pred_fake = netd_b(fake_b.detach())
                loss_d_fake = criterion_GAN(pred_fake, fake_labels)

                # Total loss
                V_d_b = (loss_d_real + loss_d_fake)*0.5
                V_d_b.backward()

                optimizer_d_b.step()

            global_i += 1
            print("===> Epoch[{}]({}/{}): V_d_a: {:.4f} V_d_b: {:.4f} V_g: {:.4f}".format(
                epoch, ii, len(dataloader), V_d_a, V_d_b, V_g
            ))

            if global_i % opt.plot_every == 0:
                time_consumed = str(round((time() - time_init) / 3600, 2)) + 'h'
                visual.img(epoch, time_consumed, fake_b.detach(), real_img.detach(), None)
                visual.loss(epoch, V_g.detach(), V_d_a.detach(), V_d_b.detach())

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_a.step()
        lr_scheduler_D_b.step()

        if (epoch + 1) % opt.save_every == 0:
            # save model
            path = opt.checkpoint_path + 'epoch' + str(epoch) + '.tar'
            t.save({
                'epoch': epoch,
                'd_a_state_dict': netd_a.state_dict(),
                'd_b_state_dict': netd_b.state_dict(),
                'g_a2b_state_dict': netg_a2b.state_dict(),
                'g_b2a_state_dict': netg_b2a.state_dict(),
                'optimizer_d_a_state_dict': optimizer_d_a.state_dict(),
                'optimizer_d_b_state_dict': optimizer_d_b.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict()
            }, path)


if __name__ == '__main__':
    train()
