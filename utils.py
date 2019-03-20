from config import *
from model import *
import torch as t
import os
import torchvision as tv
import visdom
import numpy as np

opt = Config()


class Visualize():
    def __init__(self):
        # os.system('python -m visdom.server -port 80')
        self.vis = visdom.Visdom(env=opt.visdom_env, port=80)
        # self.vis.line(X=t.zeros(1), Y=t.zeros(1), win='g', name='g', opts={'title': 'g_loss'})
        # self.vis.line(X=t.zeros(1), Y=t.zeros(1), win='d', name='d', opts={'title': 'd_loss'})
        self.vis.images(np.random.rand(4, 3, 256, 256), win='a', nrow=2, opts={'title': 'image A'})
        self.vis.images(np.random.rand(4, 3, 256, 256), win='b', nrow=2, opts={'title': 'image B'})

        self.imga = t.rand(opt.gen_num, 3, opt.image_size, opt.image_size).to(opt.device)
        self.imgb = t.rand(opt.gen_num, 3, opt.image_size, opt.image_size).to(opt.device)
        # self.vis.text()

    def img(self, epoch, time_consumed, a_fake, real_img, scores):
        # indexs = scores.topk(opt.gen_num)[1]
        result = []
        result_restore = []
        img_a_restore = []
        filename = opt.save_path + 'e' + str(epoch) + '-' + time_consumed + '.png'
        # sample_range = len(a_fake)>=4?4:1
        for each_pic in range(1):
            result.append(a_fake[each_pic])
            result_restore.append(((a_fake[each_pic] * 0.5) + 0.5) * 255)
            img_a_restore.append(((real_img[each_pic] * 0.5) + 0.5) * 255)
        tv.utils.save_image(t.stack(result), filename, normalize=True, range=(-1, 1), nrow=2)

        # t.cat(img_a_restore, out=self.imga)
        # t.cat(result_restore, out=self.imgb)
        # self.imga.reshape(opt.gen_num, 3, opt.image_size, opt.image_size)
        # self.imgb.reshape(opt.gen_num, 3, opt.image_size, opt.image_size)
        for i in range(1):
            self.imga[i] = img_a_restore[i]
            self.imgb[i] = result_restore[i]
        self.vis.images(self.imga, nrow=1, win='a', opts={'title': 'image A'})
        self.vis.images(self.imgb, nrow=1, win='b', opts={'title': 'image B'})

    def loss(self, epoch, V_g, V_d_a, V_d_b):
        epoch = t.Tensor([epoch])
        V_g, V_d, zz_loss = t.Tensor([V_g]), t.Tensor([V_d_a]), t.Tensor([V_d_b])
        self.vis.line(X=epoch, Y=V_g, update='append', win='g', opts={'title': 'V_g'})
        self.vis.line(X=epoch, Y=V_d, update='append', win='d', opts={'title': 'V_d_a'})
        self.vis.line(X=epoch, Y=zz_loss, update='append', win='zz', opts={'title': 'V_d_b'})


def upsample(input):
    return t.nn.functional.upsample(input, size=[opt.image_size, opt.image_size])


# 输入目录路径，输出最新文件完整路径
def find_new_file(dir):
    '''查找目录下最新的文件'''
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + "\\" + fn)
    if not os.path.isdir(dir + "\\" + fn) else 0)
    # print('最新的文件为： ' + file_lists[-1])
    file = os.path.join(dir, file_lists[-1])
    # print('完整路径：', file)
    return file

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
