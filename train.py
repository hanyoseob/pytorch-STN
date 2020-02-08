from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from statistics import mean


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_STN = args.lr_STN
        self.lr_CLS = args.lr_CLS

        self.wgt_STN = args.wgt_STN
        self.wgt_CLS = args.wgt_CLS

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, net_STN, net_CLS, optim_STN, optim_CLS, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'net_STN': net_STN.state_dict(), 'net_CLS': net_CLS.state_dict(),
                    'optim_STN': optim_STN.state_dict(), 'optim_CLS': optim_CLS.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, net_STN, net_CLS, optim_STN=[], optim_CLS=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            net_STN.load_state_dict(dict_net['net_STN'])
            net_CLS.load_state_dict(dict_net['net_CLS'])
            optim_STN.load_state_dict(dict_net['optim_STN'])
            optim_CLS.load_state_dict(dict_net['optim_CLS'])

            return net_STN, net_CLS, optim_STN, optim_CLS, epoch

        elif mode == 'test':
            net_STN.load_state_dict(dict_net['net_STN'])
            net_CLS.load_state_dict(dict_net['net_CLS'])

            return net_STN, net_CLS, epoch

    def preprocess(self, data):
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        normalize = Normalize()
        randomflip = RandomFlip()
        totensor = ToTensor()
        # return totensor(randomcrop(rescale(randomflip(nomalize(data)))))
        return totensor(normalize(rescale(data)))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denormalize()
        return denomalize(tonumpy(data))


    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_STN = self.lr_STN
        lr_CLS = self.lr_CLS

        wgt_STN = self.wgt_STN
        wgt_CLS = self.wgt_CLS

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data)
        dir_log = os.path.join(self.dir_log, self.scope, name_data)

        transform_train = transforms.Compose([transforms.ToTensor(), Normalize()])
        transform_inv = transforms.Compose([ToNumpy(), Denormalize()])

        dataset_train = datasets.MNIST(root='.', train=True, download=True, transform=transform_train)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

        num_train = len(loader_train.dataset)
        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        net_STN = STN(nch_in, nch_out, nch_ker, norm).to(device)
        net_CLS = CLS(nch_in, nch_out, nch_ker, norm).to(device)

        # init_net(net_STN, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(net_CLS, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        net_STN.affinement[-1].linear.weight.data.zero_()
        net_STN.affinement[-1].linear.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

        ## setup loss & optimization
        # fn_GAN = nn.BCEWithLogitsLoss().to(device)
        fn_CLS = nn.NLLLoss().to(device)

        params_STN = net_STN.parameters()
        params_CLS = net_CLS.parameters()

        optim_STN = torch.optim.Adam(params_STN, lr=lr_STN)
        optim_CLS = torch.optim.Adam(params_CLS, lr=lr_CLS)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            net_STN, net_CLS, optim_STN, optim_CLS, st_epoch = self.load(dir_chck, net_STN, net_CLS, optim_STN, optim_CLS, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            net_STN.train()
            net_CLS.train()

            loss_CLS_train = []
            pred_CLS_train = []

            # for i, data in enumerate(loader_train, 1):
            for i, (input, label) in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input = input.to(device)
                label = label.to(device)

                # forward netG
                if self.scope == 'stn':
                    input_stn = net_STN(input)
                    output = net_CLS(input_stn)
                    pred = output.max(1, keepdim=True)[1]

                    # backward netD
                    optim_STN.zero_grad()
                    optim_CLS.zero_grad()

                    loss_CLS = fn_CLS(output, label)
                    loss_CLS.backward()

                    optim_STN.step()
                    optim_CLS.step()

                elif self.scope == 'cls':
                    output = net_CLS(input)
                    pred = output.max(1, keepdim=True)[1]

                    # backward netD
                    optim_CLS.zero_grad()

                    loss_CLS = fn_CLS(output, label)
                    loss_CLS.backward()

                    optim_CLS.step()

                # get losses
                loss_CLS_train += [loss_CLS.item()]
                pred_CLS_train += [pred.eq(label.view_as(pred)).sum().item() / label.shape[0]]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: CLS: %.4f ACC: %.4f' %
                      (epoch, i, num_batch_train, mean(loss_CLS_train), 100 * mean(pred_CLS_train)))

                if should(num_freq_disp):
                    ## show output
                    input = transform_inv(input)
                    writer_train.add_images('input', input, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

                    if self.scope == 'stn':
                        input_stn = transform_inv(input_stn)
                        writer_train.add_images('input_stn', input_stn, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss_CLS', mean(loss_CLS_train), epoch)

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, net_STN, net_CLS, optim_STN, optim_CLS, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_STN = self.lr_STN
        lr_CLS = self.lr_CLS

        wgt_STN = self.wgt_STN
        wgt_CLS = self.wgt_CLS

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result = os.path.join(self.dir_result, self.scope, name_data)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        transform_test = transforms.Compose([transforms.ToTensor(), Normalize()])
        transform_inv = transforms.Compose([ToNumpy(), Denormalize()])

        dataset_test = datasets.MNIST(root='.', train=False, download=True, transform=transform_test)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        num_test = len(loader_test.dataset)
        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        net_STN = STN(nch_in, nch_out, nch_ker, norm).to(device)
        net_CLS = CLS(nch_in, nch_out, nch_ker, norm).to(device)

        # init_net(net_STN, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(net_CLS, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        net_STN.affinement[-1].linear.weight.data.zero_()
        net_STN.affinement[-1].linear.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

        ## setup loss & optimization
        # fn_GAN = nn.BCEWithLogitsLoss().to(device)
        fn_CLS = nn.NLLLoss().to(device)

        ## load from checkpoints
        st_epoch = 0

        net_STN, net_CLS, st_epoch = self.load(dir_chck, net_STN, net_CLS, mode=mode)

        ## test phase
        with torch.no_grad():
            net_STN.eval()
            net_CLS.eval()

            loss_CLS_test = []
            pred_CLS_test = []

            # for i, data in enumerate(loader_train, 1):
            for i, (input, label) in enumerate(loader_test, 1):

                input = input.to(device)
                label = label.to(device)

                # forward netG
                input_stn = net_STN(input)
                output = net_CLS(input_stn)
                pred = output.max(1, keepdim=True)[1]

                loss_CLS = fn_CLS(output, label)

                # get losses
                loss_CLS_test += [loss_CLS.item()]
                pred_CLS_test += [pred.eq(label.view_as(pred)).sum().item()/label.shape[0]]

                print('TEST: BATCH %04d/%04d: CLS: %.4f ACC: %.4f' % (i, num_batch_test, mean(loss_CLS_test), 100 * mean(pred_CLS_test)))

                ## show output
                input = transform_inv(input)
                input_stn = transform_inv(input_stn)

                for j in range(input.shape[0]):
                    name = batch_size * (i - 1) + j
                    fileset = {'name': name,
                               'input': "%04d-input.png" % name,
                               'input_stn': "%04d-input_stn.png" % name}

                    if nch_in == 3:
                        plt.imsave(os.path.join(dir_result_save, fileset['input']), input[j, :, :, :].squeeze())
                        plt.imsave(os.path.join(dir_result_save, fileset['input_stn']), input_stn[j, :, :, :].squeeze())
                    elif nch_in == 1:
                        plt.imsave(os.path.join(dir_result_save, fileset['input']), input[j, :, :, :].squeeze(), cmap='gray')
                        plt.imsave(os.path.join(dir_result_save, fileset['input_stn']), input_stn[j, :, :, :].squeeze(), cmap='gray')

                    append_index(dir_result, fileset)

            print('TEST: AVERAGE LOSS: %.6f' % (mean(loss_CLS_test)))
            print('TEST: AVERAGE ACC: %.6f' % (100 * mean(pred_CLS_test)))

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path



def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
