## from cutmodel
import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
## from nemar model
import itertools
import torch.nn.functional as F
import models.stn as stn
import torchvision.transforms as T
import os
from matplotlib import pyplot as plt
def linear_normalize(tmp):
    return (tmp - tmp.min())/(tmp.max() - tmp.min())

class CUTREGModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    The code incorporates end-to-end registration network

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser = stn.modify_commandline_options(parser, is_train)
        parser.add_argument('--CUT_mode', type=str, default="FastCUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=0.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=20.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_REG', type=float, default=100.0, help='weight for REG loss: L1(R(G(X),Y) - X)')
        parser.add_argument('--lambda_TV', type=float, default=200.0, help='weight for TV loss: TV(R(G(X),Y))')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()
        parser.set_defaults(nce_layers='0,2,4')


        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=20.0, flip_equivariance=False,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G_R', 'NCE', 'REG', 'TV']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_B_reg']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'R']
        else:  # during test time, only load G
            self.model_names = ['G', 'R']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netR = stn.define_stn(self.opt, self.opt.stn_type)
        self.crop = T.RandomCrop(size=512)


        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_R = torch.optim.Adam(itertools.chain(self.netR.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2),)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_R)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data[0].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_R_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        for i in range(5):
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            self.optimizer_D.step()

        # update G and R
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_R.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G_R = self.compute_G_R_loss()
        self.loss_G_R.backward()
        self.optimizer_G.step()
        self.optimizer_R.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()        
        

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
#         AtoB = self.opt.direction == 'AtoB'
#         self.real_A = input['A' if AtoB else 'B'].to(self.device)
#         self.real_B = input['B' if AtoB else 'A'].to(self.device)
#         self.image_paths = input['A_paths' if AtoB else 'B_paths']
        x, y = input
        self.real_A = x.to(self.device)
        self.real_B = y.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)
        warpped_images, self.deformation_field, reg_term = self.netR(self.real_A, self.real_B, apply_on=[self.fake_B])
        self.stn_reg_term = reg_term
        self.fake_B_reg = warpped_images[0]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        input_D = self.crop(torch.cat([self.fake_B.detach(), self.real_B],1))
        fake = input_D[:,:self.fake_B.shape[1],:,:]
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(input_D[:,self.fake_B.shape[1]:,:,:])
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_R_loss(self):
        """Calculate GAN and NCE loss for the generator G, and the registration loss: l1 norm and tv regularization term for net R and G"""
        fake = self.crop(self.fake_B)
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B.mean(dim=1, keepdim=True))
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        
        self.loss_REG = self.opt.lambda_REG * self.criterionL1(self.fake_B_reg, self.real_B)
        self.loss_TV = self.opt.lambda_TV * self.stn_reg_term

        self.loss_G_R = self.loss_G_GAN + loss_NCE_both + self.loss_REG + self.loss_TV
        return self.loss_G_R

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def save_images(self, epoch, test_data):
        num_img = len(test_data)
        img_dir = f'{self.opt.name}/images/'        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            print(f'Directory {img_dir} createrd')
        else:
            print(f'Directory {img_dir} already exists')  
        
        if num_img != 1:
            _, ax = plt.subplots(num_img, 5, figsize=(20, 10))
            [ax[0, i].set_title(title) for i, title in enumerate(['Source', "Translated", "Target", "Registered", "Deformation field"])]
            for i, data in enumerate(test_data):
                self.set_input(data)
                self.forward()
                source = linear_normalize(self.real_A[0].permute([1,2,0]).detach().cpu().numpy())
                translated = linear_normalize(self.fake_B[0].permute([1,2,0]).detach().cpu().numpy())
                target = linear_normalize(self.real_B[0].permute([1,2,0]).detach().cpu().numpy())
                registered = linear_normalize(self.fake_B_reg[0].permute([1,2,0]).detach().cpu().numpy())
                deformation_field = self.deformation_field[0].permute([1,2,0]).detach().cpu().numpy()
                df = np.zeros(shape=target.shape)
                df[:,:,0:2] = linear_normalize(deformation_field)
                [ax[i, j].imshow(img) for j, img in enumerate([source, translated, target, registered, df])]
                [ax[i, j].axis("off") for j in range(5)]
        else:
            _, ax = plt.subplots(1, 5, figsize=(20, 10))
            [ax[i].set_title(title) for i, title in enumerate(['Source', "Translated", "Target", "Registered", "Deformation field"])]
            for i, data in enumerate(test_data):
                self.set_input(data)
                self.forward()
                source = linear_normalize(self.real_A[0].permute([1,2,0]).detach().cpu().numpy())
                translated = linear_normalize(self.fake_B[0].permute([1,2,0]).detach().cpu().numpy())
                target = linear_normalize(self.real_B[0].permute([1,2,0]).detach().cpu().numpy())
                registered = linear_normalize(self.fake_B_reg[0].permute([1,2,0]).detach().cpu().numpy())
                deformation_field = self.deformation_field[0].permute([1,2,0]).detach().cpu().numpy()
                df = np.zeros(shape=target.shape)
                df[:,:,0:2] = linear_normalize(deformation_field)
                [ax[j].imshow(img) for j, img in enumerate([source, translated, target, registered, df])]
                [ax[j].axis("off") for j in range(5)]
        plt.savefig(f'{img_dir}/epoch={epoch}.png')
        plt.close()