from __future__ import print_function
import torch
from torch import nn
from optimizer.base_optimizer import base_optimizer
from network.loss import perceptural_loss
from util import util, logger, opt, curves
import os
import util.visualization as vs
from collections import OrderedDict
from network import GanLoss
import random
import numpy as np

log = logger.logger()


class optimizer(base_optimizer):

    def __init__(self, model, option=opt.opt()):
        super(optimizer, self).__init__()
        self._default_opt()
        self.opt.merge_opt(option)
        self._get_model(model)
        self._get_aux_nets()
        self._define_optim()
        self.writer = curves.writer(log_dir=self.opt.save_dir + '/log')
        self.global_step = 0
        self.epoch = 0
        if self.opt.continue_train:
            self.load()
        self.src_cond = 0
        self.dst_cond = 1

    def _default_opt(self):
        self.opt = opt.opt()
        self.opt.save_dir = '/checkpoints/default'
        self.opt.attr_names=['happy', 'angry', 'sad', 'contemptuous', 'disgusted', 'neutral', 'fearful', 'surprised']
        self.opt.dst_attr = 'neutral'

    def set_input(self, input):
        self.image, self.dst_img, self.attribute = input
        self.image = util.toVariable(self.image).cuda()
        self.dst_img = util.toVariable(self.dst_img).cuda()
        self.attribute = [util.toVariable(att.cuda()) for att in self.attribute]
        self.attribute_express = self.attribute[0]

    def zero_grad(self):
        self.G.zero_grad()
        self.domain.zero_grad()
        self.classifier.zero_grad()
        self.star_gan.zero_grad()

    def _get_aux_nets(self):
        self.perceptural_loss = perceptural_loss().cuda()

        self.batch_norm_list1 = [nn.BatchNorm2d(256, affine=False).cuda(),
                                 nn.BatchNorm2d(512, affine=False).cuda(),
                                 nn.BatchNorm2d(512, affine=False).cuda()]
        self.batch_norm_list2 = [nn.BatchNorm2d(256, affine=False).cuda(),
                                 nn.BatchNorm2d(512, affine=False).cuda(),
                                 nn.BatchNorm2d(512, affine=False).cuda()]

    def _get_model(self, model):
        model = [tmp.cuda() for tmp in model]
        self.star_gan, self.G, self.domain, self.classifier = model

        self.print_network(self.star_gan, 'star_gan')
        self.print_network(self.G, 'G')
        self.print_network(self.domain, 'Domain')
        self.print_network(self.classifier, 'classifier')


    def _define_optim(self):
        self.optim_G = torch.optim.Adam(self.G.parameters(), lr=self.opt.Glr, betas=[0.5, 0.999])
        self.optim_domain = torch.optim.Adam(self.domain.parameters(), lr=self.opt.Dlr, betas=[0.5, 0.999])
        self.optim_classifier = torch.optim.Adam(self.classifier.parameters(), lr=self.opt.Glr, betas=[0.5, 0.999])

    def load(self, label='latest', save_dir=None):
        if save_dir is None:
            save_dir = self.opt.save_dir + '/{}-{}.pth'
        else:
            save_dir = save_dir + '/{}-{}.pth'

        self._check_and_load(self.G, save_dir.format('G', label))
        self._check_and_load(self.domain, save_dir.format('Domain', label))
        self._check_and_load(self.classifier, save_dir.format('domain_attribute', label))

        self._check_and_load(self.optim_G, save_dir.format('optim_G', label))
        self._check_and_load(self.optim_domain, save_dir.format('optim_domain', label))
        self._check_and_load(self.optim_classifier, save_dir.format('optim_classifier', label))


    def save(self, label='latest'):
        save_dir = self.opt.save_dir + '/{}-{}.pth'
        torch.save(self.G.state_dict(), save_dir.format('G', label))
        torch.save(self.domain.state_dict(), save_dir.format('Domain', label))
        torch.save(self.classifier.state_dict(), save_dir.format('domain_attribute', label))
        torch.save(self.optim_G.state_dict(), save_dir.format('optim_G', label))
        torch.save(self.optim_domain.state_dict(), save_dir.format('optim_domain', label))
        torch.save(self.optim_classifier.state_dict(), save_dir.format('optim_classifier', label))

    def optimize_parameters(self, global_step):
        '''
        in the following code, we denote A as source domain, B as target domain.
        The mapping network between A and B is self.G.
        Note that self.G is a conditional network. with different conditions (self.src_cond or self.dst_cond), we
        actually call different functions.
        The discriminator is self.domain. It is also a similar conditional network.

        '''
        self.G.train()
        self.domain.train()
        self.star_gan.eval()
        self.classifier.train()
        self.loss = OrderedDict()

        self.ganloss = GanLoss.LSGANLoss_multiScale()
        self.cls_loss = nn.CrossEntropyLoss().cuda()

        self.ones = util.toVariable(torch.ones(self.image.size(0))).cuda()
        self.zeros = util.toVariable(torch.zeros(self.image.size(0))).cuda()

        self.img_BA = self.G(self.dst_img, self.src_cond, self.dst_img)
        self.c_trg = self.generate_label_vertor()
        self.img_BAG = self.star_gan(self.img_BA, self.c_trg)
        self.img_BAGB = self.G(self.img_BAG, self.dst_cond, self.dst_img)

        self.zero_grad()
        self.compute_domain_Loss().backward(retain_graph=True)
        self.optim_domain.step()

        self.zero_grad()
        self.compute_cls_loss().backward()
        self.optim_classifier.step()

        if global_step % self.opt.n_dis == 0:
            self.zero_grad()
            self.compute_G_loss().backward()
            self.optim_G.step()

    def compute_cls_loss(self):
        c_trg = self.generate_label_vertor()
        img_AG = self.star_gan(self.image, c_trg)
        feature_source, source_attribute = self.classifier(img_AG)

        c_trg_label = torch.max(c_trg, dim=1)[1]
        loss_class_src = self.cls_loss(source_attribute, c_trg_label.long())
        self.loss['classify'] = loss_class_src

        return self.loss['classify']

    def compute_domain_Loss(self):
        self.loss['domain'] = 0
        img_BA = self.img_BA.detach()
        domain_A_real = self.domain(self.image, self.src_cond)
        domain_BA_fake = self.domain(img_BA.detach(), self.src_cond)
        self.loss['domain-A'] = self.ganloss(domain_BA_fake, domain_A_real, 'd') * self.opt.lam_gan
        self.loss['domain'] += self.loss['domain-A']
        return self.loss['domain']

    def compute_G_loss(self):
        self.loss['G'] = 0
        img_BA = self.img_BA
        img_BAG = self.img_BAG
        img_BAGB = self.img_BAGB
        c_trg = self.c_trg.clone()

        ''' gan loss '''
        img_BA_disc = self.domain(img_BA, self.src_cond)
        self.loss['G-BA-fake'] = self.ganloss(img_BA_disc, None, 'g') * self.opt.lam_gan
        self.loss['G'] += self.loss['G-BA-fake']

        ''' classification loss for BAG '''
        _, img_BAG_label = self.classifier(img_BAG)
        c_trg_label = torch.max(c_trg.clone(), dim=1)[1]
        loss_stargan_attribute = self.cls_loss(img_BAG_label, c_trg_label.long())
        self.loss['G-cls'] = loss_stargan_attribute * self.opt.lam_cls
        self.loss['G'] += self.loss['G-cls']

        ''' recon loss '''
        if self.opt.recon_type == 'BAB':
            img_recon = self.G(img_BA, self.dst_cond, self.dst_img)
        elif self.opt.recon_type == 'BAGB':
            dst_label = self.label2onehot(torch.ones(self.image.size(0)) * self.opt.attr_names.index(self.opt.dst_attr), len(self.opt.attr_names)).cuda()
            img_BAG_dst_label = self.star_gan(img_BA, dst_label)
            img_recon = self.G(img_BAG_dst_label, self.dst_cond, self.dst_img)
        else:
            raise NotImplementedError
        if self.opt.lam_recon > 0:
            self.loss['G-BA-recon'] = nn.L1Loss()(img_recon, self.dst_img.detach()) * self.opt.lam_recon
            self.loss['G'] += self.loss['G-BA-recon']
        if self.opt.lam_recon_per > 0:
            self.loss['G-BA-recon-per'] = self.perceptural_loss(img_recon, self.dst_img.detach()) * self.opt.lam_recon_per
            self.loss['G'] += self.loss['G-BA-recon-per']

        ''' analogy loss '''
        if self.opt.lam_vgg > 0:
            self.loss['G-vgg'] = self.vgg_loss(img_BA, img_BAG, self.dst_img, img_BAGB,
                                            self.opt.vgg_layers, self.opt.analogy_scale) * self.opt.lam_vgg
            self.loss['G'] += self.loss['G-vgg']
        return self.loss['G']

    def vgg_loss(self, A, A_, B, B_, layers, scale):
        loss = 0
        all_vgg_A = self.perceptural_loss.vgg(A)
        all_vgg_A_ = self.perceptural_loss.vgg(A_)
        all_vgg_B = self.perceptural_loss.vgg(B)
        all_vgg_B_ = self.perceptural_loss.vgg(B_)
        for l in layers:
            vgg_A = all_vgg_A[l]
            vgg_A_ = all_vgg_A_[l]
            vgg_B = all_vgg_B[l]
            vgg_B_ = all_vgg_B_[l]

            _ = self.batch_norm_list1[l](vgg_A)
            _ = self.batch_norm_list1[l](vgg_A_)
            _ = self.batch_norm_list2[l](vgg_B)
            _ = self.batch_norm_list2[l](vgg_B_)
            self.batch_norm_list1[l].eval()
            self.batch_norm_list2[l].eval()

            vgg_A = self.batch_norm_list1[l](vgg_A)
            vgg_A_ = self.batch_norm_list1[l](vgg_A_)
            vgg_B = self.batch_norm_list2[l](vgg_B)
            vgg_B_ = self.batch_norm_list2[l](vgg_B_)

            shift_A = vgg_A_ - vgg_A
            shift_A *= scale
            shift_B = vgg_B_ - vgg_B
            loss += (shift_B - shift_A).abs().mean()
            self.batch_norm_list1[l].train()
            self.batch_norm_list2[l].train()
        return loss

    def generate_label_vertor(self):
        selected_vector = []
        for i in range(self.image.size(0)):
            tmp = torch.randperm(len(self.opt.attr_names))[0]
            one_hot_vec = torch.zeros(1, len(self.opt.attr_names))
            one_hot_vec[:, tmp] = 1
            selected_vector += [one_hot_vec]
        selected_vector = torch.cat(selected_vector, dim=0)
        selected_vector = util.toVariable(selected_vector).cuda()
        return selected_vector

    def get_current_errors(self):
        return self.loss

    def save_samples(self, global_step=0):
        with torch.no_grad():
            self.save_samples_dst(global_step)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def save_samples_dst(self, global_step):
        self.G.eval()
        self.star_gan.eval()
        n_branches = len(self.opt.attr_names)
        save_path = os.path.join(self.opt.save_dir, 'dst_samples')
        util.mkdir(save_path)
        img_all = [self.dst_img.data.cpu()]

        random_index = random.randint(0, n_branches - 1)
        c_trg_dst = self.label2onehot(torch.ones(self.image.size(0)) * random_index, n_branches).cuda()
        change_dst_im = self.star_gan(self.dst_img, c_trg_dst)
        img_all += [change_dst_im.data.cpu()]

        img_BA = self.G(self.dst_img, self.src_cond, self.dst_img)
        img_all += [img_BA.data.cpu()]
        img_all += [self.G(img_BA, self.dst_cond, self.dst_img).data.cpu()]

        for i in range(n_branches):
            c_trg = self.label2onehot(torch.ones(self.image.size(0)) * i, n_branches).cuda()

            img_BAG = self.star_gan(img_BA, c_trg)
            img_BAGB = self.G(img_BAG, self.dst_cond, self.dst_img)

            img_all += [img_BAG.data.cpu()]
            img_all += [img_BAGB.data.cpu()]
        img_all += [self.image.data.cpu()]

        random_index = random.randint(0, n_branches-1)
        c_trg = self.label2onehot(torch.ones(self.image.size(0)) * random_index, n_branches).cuda()
        change_im = self.star_gan(self.image, c_trg)
        img_all += [change_im.data.cpu()]

        img_all = torch.cat(img_all, dim=0)
        img_all = vs.untransformTensor(img_all.cpu())
        vs.writeTensor('%s/%d.jpg' % (save_path, global_step), img_all, nRow=self.image.size(0))


