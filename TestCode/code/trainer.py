import os
import math
from decimal import Decimal
from PIL import Image
import utility
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from skimage.measure import compare_psnr
from torchvision import transforms
from torchvision.transforms.functional import rotate, hflip, vflip, to_tensor, to_pil_image
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    #Karim 159773
                    sr1 = self.model(lr, idx_scale) #original
                    #sr = utility.enhance_Precision(self.model, lr, idx_scale)
                    sr2 = self.model(lr.transpose(2,3), idx_scale).transpose(2,3) #90
                    sr3 = self.model(lr.flip(2), idx_scale).flip(2) #180
                    sr4 = self.model(lr.transpose(2, 3).flip(3), idx_scale).flip(3).transpose(2, 3) #270
                    sr5 = self.model(lr.flip(3), idx_scale).flip(3) # original flipped
                    sr6 = self.model(lr.transpose(2,3).flip(2), idx_scale).flip(2).transpose(2,3) #90 flipped
                    sr7 = self.model(lr.flip(2).flip(3), idx_scale).flip(3).flip(2) #180 flipped
                    sr8 = self.model(lr.transpose(2, 3).flip(3).flip(2), idx_scale).flip(2).flip(3).transpose(2, 3) #270 flipped
                    #sr = sr.transpose(2,3);
                    sr = torch.div(
                      torch.add(
                        torch.add(
                          torch.add(
                            torch.add(
                              torch.add(
                                torch.add(
                                  torch.add(
                                    sr1, sr2
                                    ), 
                                    sr3), 
                                    sr4),
                                    sr5),
                                    sr6),
                                    sr7),
                                    sr8), 
                                    8)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        #fileName = filename.split('_')[0]
                        #print("\n\nFileName: ", fileName, "\n\n")
                        #pic_test = np.array(Image.open("../SR/BI/SAN_159773/Set5/x2/" + fileName + "_LRBI_x2.png"))
                        #pic_real = np.array(Image.open("../HR/Set5/x2/" + fileName + "_HR_x2.png"))
                        #print("\n\nHR shape: ", pic_real.shape, "\n\n")
                        #print("\n\nSR shape: ", pic_test.shape, "\n\n")
                        #eval_acc = compare_psnr(pic_real, pic_test, self.args.rgb_range)
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        #self.ckp.save_results(filename, save_list, scale)
                        self.ckp.save_results_nopostfix(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s, ave time: {:.2f}s\n'.format(timer_test.toc(), timer_test.toc()/len(self.loader_test)), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs


