import os
import math
import time
import datetime
from functools import reduce
from PIL import Image
from torchvision.utils import save_image
from numpy import fliplr, flipud
from torchvision.transforms.functional import rotate, hflip, vflip, to_tensor, to_pil_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import scipy.misc as misc
from tensorflow import convert_to_tensor
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../SR/' + args.degradation + '/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        
        _make_dir(self.dir + '/' + args.testset + '/x' + str(args.scale[0]))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}{}.png'.format(filename, p), ndarr)

    def save_results_nopostfix(self, filename, save_list, scale):
        #print(filename)
        if self.args.degradation == 'BI':
            filename = filename.replace("LRBI", self.args.save)
        elif self.args.degradation == 'BD':
            filename = filename.replace("LRBD", self.args.save)
        
        filename = '{}/{}/x{}/{}'.format(self.dir, self.args.testset, scale, filename)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}.png'.format(filename), ndarr)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

#Karim 159773
def enhance_Precision(model, lr, idx_scale):

    #lr = transforms.ToPILImage()(lr.cpu().squeeze_(0))
    
    print("\n\nLR: ", lr, "\n\n");
    save_image(lr, 'lr.jpg')

    #print("\n\nType lr: ", type(lr), "\n\n")
    lr90 = rotate(lr, 90);

    lr180 = rotate(lr, 180);
    lr270 = rotate(lr, 270);
    #lr90 = convert_to_tensor(lr90)
    #print("\n\nType lr90: ", type(lr90), "\n\n")

    flippedlr_lr90 = hflip(lr90);

    flippedlr_lr180 = hflip(lr180);
    flippedlr_lr270 = hflip(lr270);
    #print("\n\nType flippedlr_lr90: ", type(flippedlr_lr90), "\n\n")

    flippedud_lr90 = vflip(lr90);
    flippedud_lr180 = vflip(lr180);
    flippedud_lr270 = vflip(lr270);
    #print("\n\nType flippedlr_lr90: ", type(flippedud_lr90), "\n\n")
    
    #lr90 = (transforms.ToTensor()(lr90).unsqueeze_(0)).cuda()
    #lr90 = to_tensor(lr90).unsqueeze_(0).cuda()
    lr90 = transforms.ToPILImage()(model(lr90, idx_scale).cpu().squeeze_(0))
    lr90.save("lr90.jpg")
    lr180 = (transforms.ToTensor()(lr180).unsqueeze_(0)).cuda()
    lr180 = transforms.ToPILImage()(model(lr180, idx_scale).cpu().squeeze_(0))

    lr270 = (transforms.ToTensor()(lr270).unsqueeze_(0)).cuda()
    lr270 = transforms.ToPILImage()(model(lr270, idx_scale).cpu().squeeze_(0))

    flippedlr_lr90 = (transforms.ToTensor()(flippedlr_lr90).unsqueeze_(0)).cuda()
    flippedlr_lr90 = transforms.ToPILImage()(model(flippedlr_lr90, idx_scale).cpu().squeeze_(0))
    #flippedlr_lr90 = Image.fromarray(flippedlr_lr90)
    flippedlr_lr90.save("flippedlr_lr90.jpg")
    flippedlr_lr180 = (transforms.ToTensor()(flippedlr_lr180).unsqueeze_(0)).cuda()
    flippedlr_lr180 = transforms.ToPILImage()(model(flippedlr_lr180, idx_scale).cpu().squeeze_(0))
    
    flippedlr_lr270 = (transforms.ToTensor()(flippedlr_lr270).unsqueeze_(0)).cuda()
    flippedlr_lr270 = transforms.ToPILImage()(model(flippedlr_lr270, idx_scale).cpu().squeeze_(0))
    
    flippedud_lr90 = (transforms.ToTensor()(flippedud_lr90).unsqueeze_(0)).cuda()
    flippedud_lr90 = transforms.ToPILImage()(model(flippedud_lr90, idx_scale).cpu().squeeze_(0))
    
    flippedud_lr180 = (transforms.ToTensor()(flippedud_lr180).unsqueeze_(0)).cuda()
    flippedud_lr180 = transforms.ToPILImage()(model(flippedud_lr180, idx_scale).cpu().squeeze_(0))
        
    flippedud_lr270 = (transforms.ToTensor()(flippedud_lr270).unsqueeze_(0)).cuda()
    flippedud_lr270 = transforms.ToPILImage()(model(flippedud_lr270, idx_scale).cpu().squeeze_(0))
    
    sr90 = rotate(lr90, -90);
    sr180 = rotate(lr180, -180);
    sr270 = rotate(lr270, -270);
    sr90.save("sr90.jpg")
    #transforms.ToPILImage()(lr.cpu().squeeze_(0))
    srflr_lr90 = rotate(hflip(flippedlr_lr90), -90);
    srflr_lr180 = rotate(hflip(flippedlr_lr180), -180);
    srflr_lr270 = rotate(hflip(flippedlr_lr270), -270);
    sr90.save("srflr_lr90.jpg")
    srfud_lr90 = rotate(vflip(flippedud_lr90), -90);
    srfud_lr180 = rotate(vflip(flippedud_lr180), -180);
    srfud_lr270 = rotate(vflip(flippedud_lr270), -270);
    
    sr = np.array(sr90, dtype=np.float32 ) / 9;
    sr += np.array(sr180, dtype=np.float32 ) / 9;
    sr += np.array(sr270, dtype=np.float32 ) / 9;
    sr += np.array(srflr_lr90, dtype=np.float32 ) / 9;
    sr += np.array(srflr_lr180, dtype=np.float32 ) / 9;
    sr += np.array(srflr_lr270, dtype=np.float32 ) / 9;
    sr += np.array(srfud_lr90, dtype=np.float32 ) / 9;
    sr += np.array(srfud_lr180, dtype=np.float32 ) / 9;
    sr += np.array(srfud_lr270, dtype=np.float32 ) / 9;

    SR = np.array(np.round(sr), dtype=np.uint8)
    #SR = Image.fromarray(arr,mode="RGB")
    SR = (transforms.ToTensor()(transforms.ToPILImage()(SR)).unsqueeze_(0)).cuda()
    return SR

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

