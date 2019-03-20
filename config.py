import torch as t


class Config():
    num_workers = 0  # 多进程加载数据所用的进程数
    image_size = 256

    # optimizer
    lr = 0.0002
    beta1 = 0.5
    weight_decay = 1e-4
    n_epochs = 200

    max_epoch = 200
    batch_size = 1
    lambda_g = 1
    lambda_c = 10

    use_gpu = True
    device = t.device('cuda:0')
    d_every = 1  # train d per x batch
    g_every = 1  # train g per x batch

    save_path = 'imgs/'
    # true_imgs_path = 'minidataset/true_imgs'
    # style_imgs_path = 'minidataset/style_imgs'
    true_imgs_path = 'mydataset/true_imgs'
    style_imgs_path = 'mydataset/style_imgs'
    checkpoint_path = 'checkpoints/'

    plot_every = 50
    save_every = 10
    visdom_env = '0'
    gen_img = 'imgs/result.png'
    gen_num = 4

    debug = False

