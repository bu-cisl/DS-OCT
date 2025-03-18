## from cutmodel
import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import util.metrics_utils as M
## from nemar model
import itertools
import torch.nn.functional as F
import torchvision.transforms as T
import models.stn as stn
import os
from matplotlib import pyplot as plt
def linear_normalize(tmp):
    return (tmp - tmp.min())/(tmp.max() - tmp.min())
from util.data_preparation import patch_wise_predict, rgb_to_lab, lab_to_rgb, preprocess_lab, deprocess_lab


class CUTREGtwostageModel(BaseModel):
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

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))') ## 1.0
        parser.add_argument('--lambda_NCE', type=float, default=10.0, help='weight for NCE loss: NCE(G(X), X)') ## 10.0
        parser.add_argument('--lambda_REG', type=float, default=100.0, help='weight for REG loss: L1(R(G(X),Y) - X)') # 100 
        parser.add_argument('--lambda_TV', type=float, default=600.0, help='weight for TV loss: TV(R(G(X),Y))') # 200 x 3=600
        parser.add_argument('--lambda_MAE', type=float, default=10.0, help='weight for MAE loss: TV(R(G(X),Y))')
        parser.add_argument('--lambda_MAE_pseudo', type=float, default=10.0, help='weight for MAE_pseudo loss: TV(R(G(X),Y))')
        parser.add_argument('--LAB_space', type=util.str2bool, default=False, help='whether to convert GT to LAB color space')
        parser.add_argument('--lambda_LAB', type=float, default=0.0, help='weight for color loss:')

        parser.add_argument('--train_R_with_G', type=util.str2bool, nargs='?', const=True, default=True, help='train registration network with loss terms dependent on generator')
        parser.add_argument('--only_train_R', type=util.str2bool, nargs='?', const=True, default=False, help='only train registration network')
        parser.add_argument('--only_train_G', type=util.str2bool, nargs='?', const=True, default=False, help='only train generator by CUT')
        parser.add_argument('--train_G_pseudo', type=util.str2bool, nargs='?', const=True, default=False, help='only train generator by pseudo')


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

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0, flip_equivariance=False,
                n_epochs=50, n_epochs_decay=150)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=20.0, flip_equivariance=False,
                n_epochs=50, n_epochs_decay=150
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'REG', 'TV', 'MAE', 'MAE_pseudo', 'LAB']
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
        self.mini_bs = 1


        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionNCE = []
            self.criterionL2 = torch.nn.MSELoss()
            self.NLCC = stn.stn_losses.NLCC()

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
        self.define_fold(data)
        x, y = data[0][0].repeat(1,3,1,1), data[1][0]

        real_A_patches = self.unfold(x.permute(1,0,2,3)).reshape(self.cx,self.dim,self.dim,-1).permute(3,0,1,2)
        real_B_patches = self.unfold(y.permute(1,0,2,3)).reshape(self.cy,self.dim,self.dim,-1).permute(3,0,1,2)

        bs_per_gpu = self.mini_bs // max(len(self.opt.gpu_ids), 1)
        self.real_A = real_A_patches[:bs_per_gpu].to(self.device)
        self.real_A_eq = real_A_patches[:bs_per_gpu].to(self.device)
        self.fake_A_eq = real_A_patches[:bs_per_gpu].to(self.device)
        self.real_B = real_B_patches[:bs_per_gpu].to(self.device)
        self.real_B_reg = real_B_patches[:bs_per_gpu].to(self.device)

        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        if self.opt.lambda_GAN > 0.0:
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if (self.opt.netF == 'mlp_sample') & (self.opt.lambda_NCE > 0.0):
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if (self.opt.netF == 'mlp_sample') & (self.opt.lambda_NCE > 0.0):
            self.optimizer_F.step()        

    def optimize_parameters_stage2(self, fake_B = None):
        # forward
        self.forward_stage2(fake_B = fake_B)
        # update R
        self.optimizer_R.zero_grad()
        self.loss_R = self.compute_R_loss()
        self.loss_R.backward()
        self.optimizer_R.step()


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
        data = input
        x, y = data[0][0], data[1][0]
        
        self.real_A = x.to(self.device)
        self.real_B = y.to(self.device)
        """perform histogram normalization on OCT and Gallyas OD whole slide image"""
        self.real_A_eq = (T.functional.equalize((255*self.real_A).type(torch.uint8))/255).repeat(1,3,1,1)
        self.fake_A = torch.clamp(- torch.log10(self.real_B).mean(dim=1, keepdim=True),min=0,max=1)
        self.fake_A_eq = (T.functional.equalize((255*(1.0 - self.fake_A)).type(torch.uint8))/255).repeat(1,3,1,1)
        '''perform RGB to LAB color space transfer'''
        if self.opt.LAB_space:
            self.real_B = preprocess_lab(rgb_to_lab(self.real_B.permute(0,2,3,1))).permute(0,3,1,2).to(self.device)
        
    def patchify_WSI(self, real_B_reg=None):
        real_A_eq_patches = self.unfold(self.real_A_eq.permute(1,0,2,3)).reshape(self.cx,self.dim,self.dim,-1).permute(3,0,1,2).detach().cpu()
        if real_B_reg != None:
            real_B_reg_patches = self.unfold(real_B_reg.permute(1,0,2,3)).reshape(self.cy,self.dim,self.dim,-1).permute(3,0,1,2).detach().cpu()
        else:
            real_B_reg_patches = self.unfold(self.real_B.permute(1,0,2,3)).reshape(self.cy,self.dim,self.dim,-1).permute(3,0,1,2).detach().cpu()
            
        if self.opt.train_G_pseudo:
            fake_A_eq_patches = self.unfold(self.fake_A_eq.permute(1,0,2,3)).reshape(self.cx,self.dim,self.dim,-1).permute(3,0,1,2).detach().cpu()
            real_B_patches = self.unfold(self.real_B.permute(1,0,2,3)).reshape(self.cy,self.dim,self.dim,-1).permute(3,0,1,2).detach().cpu()
            mini_dataset = torch.utils.data.TensorDataset(real_A_eq_patches, real_B_reg_patches, fake_A_eq_patches, real_B_patches)
        else:
            mini_dataset = torch.utils.data.TensorDataset(real_A_eq_patches, real_B_reg_patches)
        mini_dataloader = torch.utils.data.DataLoader(mini_dataset, batch_size=self.mini_bs, shuffle=False) 
        return mini_dataloader
    
    def set_mini_input(self, input):
        if len(input) == 4 :
            x1, y1, x2, y2 = input
            self.real_A_eq = x1.to(self.device)
            self.real_B_reg = y1.to(self.device)
            self.fake_A_eq = x2.to(self.device)
            self.real_B = y2.to(self.device)
        else:        
            x, y = input
            self.real_A_eq = x.to(self.device)
            self.real_B_reg = y.to(self.device)
        
    def define_fold(self, data, dim = 512):
        self.dim = dim
        x, y = data[0][0], data[1][0]
        self.cx,h,w,self.cy  = y.size(1), x.size(2), x.size(3), y.size(1)
        padding = (int(np.ceil(h/self.dim)*self.dim-h)//2,int(np.ceil(w/self.dim)*self.dim-w)//2)
        self.unfold = torch.nn.Unfold(kernel_size=(self.dim,self.dim),padding=padding,stride=self.dim)
        self.fold = torch.nn.Fold(output_size=(h,w),kernel_size=(self.dim,self.dim),padding=padding,stride=self.dim)
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A_eq)
        if self.opt.train_G_pseudo and self.opt.isTrain:
            self.rec_B = self.netG(self.fake_A_eq)
        if self.opt.nce_idt:
            self.idt_B = self.netG(self.real_B)

        
    def forward_stage2(self, fake_B=None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        warpped_images, self.deformation_field, reg_term = self.netR(self.real_A_eq, self.real_B, apply_on=[self.real_B, self.fake_A_eq])

        self.stn_reg_term = reg_term
        self.real_B_reg = warpped_images[0]
        self.fake_A_reg_eq = warpped_images[1]
        if fake_B != None:
            self.fake_B = fake_B.to(self.device)        
                
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
    
    def compute_R_loss(self):
        """Calculate loss for the registration network R: the registration loss - l1 norm and tv regularization term for net R """
        if self.opt.lambda_REG > 0.0:
            if self.opt.train_R_with_G:
                self.loss_REG = self.opt.lambda_REG * (self.criterionL1(self.fake_B, self.real_B_reg)* 0.1 +  
                                                       self.criterionL1(self.real_A_eq, self.fake_A_reg_eq)* 0.5)
            else:
                self.loss_REG = self.opt.lambda_REG * (self.criterionL1(self.real_A_eq, self.fake_A_reg_eq))* 0.5                 
   
        else:
            self.loss_REG = 0.0
        if self.opt.lambda_TV > 0.0:
            self.loss_TV = self.opt.lambda_TV * self.stn_reg_term
        else:
            self.loss_TV = 0.0
        self.loss_R = self.loss_REG + self.loss_TV
        return self.loss_R
    
    def compute_G_loss(self):
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A_eq, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
            
        if self.opt.lambda_MAE > 0.0:
            self.loss_MAE = self.opt.lambda_MAE * (self.criterionL1(fake, self.real_B_reg))
        else:
            self.loss_MAE = 0.0
            
        if self.opt.lambda_MAE_pseudo > 0.0:
            self.loss_MAE_pseudo = self.opt.lambda_MAE_pseudo * (self.criterionL1(self.rec_B, self.real_B))
        else:
            self.loss_MAE_pseudo = 0.0
        
        if self.opt.lambda_LAB > 0.0:
            real_LAB = (rgb_to_lab(self.real_B_reg.permute(0,2,3,1))).to(self.device)
            fake_LAB = (rgb_to_lab(fake.permute(0,2,3,1))).to(self.device)
            self.loss_LAB = self.opt.lambda_LAB * (self.criterionL1(real_LAB.mean(dim=(1,2)),fake_LAB.mean(dim=(1,2))) +
                                                   self.criterionL1(real_LAB.std(dim=(1,2)),fake_LAB.std(dim=(1,2))))
        else:
            self.loss_LAB = 0.0
            
        self.loss_TV = 0.0
        self.loss_REG = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_NCE + self.loss_MAE + self.loss_MAE_pseudo +self.loss_LAB
        return self.loss_G
    

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
    
    def patch_wise_predict(self, input_data, patch_dim=512, bs=2, stride_ratio=4):
        dim = patch_dim
        stride = dim//stride_ratio
        X = input_data
        self.netG.eval()
        cx,h,w,cy  = X.shape[1], X.shape[2], X.shape[3],3
        padding = (int(np.ceil(h/dim)*dim-h)//2,int(np.ceil(w/dim)*dim-w)//2)
        unfold = torch.nn.Unfold(kernel_size=(dim,dim),padding=padding,stride=stride)
        fold = torch.nn.Fold(output_size=(h,w),kernel_size=(dim,dim),padding=padding,stride=stride)
        real_A_patches = unfold(X.permute(1,0,2,3)).reshape(cx,dim,dim,-1).permute(3,0,1,2)
        test_dataset = torch.utils.data.TensorDataset(real_A_patches)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
        test_patches = torch.Tensor(np.ndarray(shape=(real_A_patches.shape[0],real_A_patches.shape[2], 
                                     real_A_patches.shape[3],3)))
        weight_patches = torch.ones_like(test_patches)
        for i, data in enumerate(test_dataloader):
            test_patches[bs*i:bs*(i+1),:,:,:] = self.netG(data[0].to(self.device)).permute([0,2,3,1]).detach().cpu()
        test = fold(test_patches.permute(3,1,2,0).reshape(cy,dim*dim,-1)).permute(1,2,3,0)
        weight = fold(weight_patches.permute(3,1,2,0).reshape(cy,dim*dim,-1)).permute(1,2,3,0)
        test = test / weight
        if self.opt.LAB_space:
            test = lab_to_rgb(deprocess_lab(test))
        return test.squeeze(0).detach()

    def save_images_FFOV(self, epoch, test_dataloader):
        num_img = 2
        metrics = np.zeros((12, num_img))
        img_dir = f'Results/{self.opt.name}/images/'        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            print(f'Directory {img_dir} created')
        else:
            print(f'Directory {img_dir} already exists')  
            
        if self.opt.only_train_G:
            _, ax = plt.subplots(num_img, 3, figsize=(40, 40))
            [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Translated", "Target"])]
        elif self.opt.only_train_R:
            _, ax = plt.subplots(num_img, 4, figsize=(40, 40))
            [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Target", "Registered", "Deformation field"])]
        elif self.opt.train_G_pseudo:
            _, ax = plt.subplots(num_img, 5, figsize=(40, 40))
            [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Translated", "Target", "Source_pseudo", "Translated_pseudo"])]            
        else:
            _, ax = plt.subplots(num_img, 5, figsize=(40, 40))
            [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Translated", "Target", "Registered", "Deformation field"])]
            
        for i, data in enumerate(test_dataloader):
            self.netG.eval()
            self.set_input(data)
            self.define_fold(data)
            if self.opt.LAB_space:
                target = lab_to_rgb(deprocess_lab(self.real_B.permute(0,2,3,1)))[0].detach().cpu().numpy()
            else:
                target = (self.real_B.permute(0,2,3,1))[0].detach().cpu().numpy()
            source = self.real_A_eq[0].permute([1,2,0]).detach().cpu().numpy()
            if self.opt.train_G_pseudo:    
                source_pseudo = self.fake_A_eq[0].permute([1,2,0]).detach().cpu().numpy()                   
            
            if not self.opt.only_train_R:
                translated = self.patch_wise_predict(input_data=self.real_A_eq, stride_ratio=1).cpu().numpy()
                translated = linear_normalize(translated)   
                if self.opt.train_G_pseudo:
                    translated_pseudo = self.patch_wise_predict(input_data=self.fake_A_eq, stride_ratio=1).cpu().numpy()    
                    translated_pseudo = linear_normalize(translated_pseudo)

            if not self.opt.only_train_G:
                self.set_input(data)
                self.forward_stage2()                    
                deformation_field = self.deformation_field[0].permute([1,2,0]).detach().cpu().numpy()                
                df = np.zeros(shape=target.shape)
                df[:,:,0:2] = linear_normalize(deformation_field)
                if self.opt.only_train_R:
                    source = self.real_A_eq[0].permute([1,2,0]).detach().cpu().numpy() 
                    target = self.fake_A_eq[0].permute([1,2,0]).detach().cpu().numpy()                
                    registered = linear_normalize(self.fake_A_reg_eq[0].permute([1,2,0]).detach().cpu().numpy())
                else:
                    registered = linear_normalize(self.real_B_reg[0].permute([1,2,0]).detach().cpu().numpy())

            if self.opt.only_train_G:
                [ax[i,j].imshow(img) for j, img in enumerate([source, translated, target])]
                [ax[i,j].axis("off") for j in range(3)]
            elif self.opt.only_train_R:
                [ax[i,j].imshow(img) for j, img in enumerate([source, target, registered, df])]
                [ax[i,j].axis("off") for j in range(4)] 
            elif self.opt.train_G_pseudo:
                [ax[i,j].imshow(img) for j, img in enumerate([source, translated, target, source_pseudo, translated_pseudo])]
                [ax[i,j].axis("off") for j in range(5)]                
            else:
                [ax[i,j].imshow(img) for j, img in enumerate([source, translated, target, registered, df])]
                [ax[i,j].axis("off") for j in range(5)]
                
#             metrics[0:4, i] = M.pixel_level_metrics(translated, registered)
#             metrics[4:6, i] = M.fiber_PCC_and_layer_IOU(translated, registered)
#             metrics[6, i] = M.vessel_IOU(translated, registered)
#             metrics[7:11, i] = M.unpaired_fiber_metrics(translated, target)
#             metrics[11, i] = M.vessel_area_diff(translated, target)

        plt.savefig(f'{img_dir}/epoch={epoch}.png')
        plt.close()
        return metrics.mean(1)