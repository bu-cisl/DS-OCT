import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import os
from matplotlib import pyplot as plt
from util.data_preparation import patch_wise_predict

import torchvision.transforms as T 
import sys
sys.path.append('/projectnb/rise2019/yiw445/Torch_version/')
from util.util import str2bool
import models.stn

import util.metrics_utils as M
def linear_normalize(tmp):
    return (tmp - tmp.min())/(tmp.max() - tmp.min())
#     return np.clip((tmp - tmp.min()) / (tmp.max() - tmp.min()), 0, 1)

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--threshold_A', type=float, default=0.6933594, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--threshold_B', type=float, default=0.5683594, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--LAB_space', type=str2bool, default=False, help='whether to convert GT to LAB color space')

            #new metrics watcher argument
#             parser.add_argument('--train_R_with_G', type=str2bool, nargs='?', const=True, default=True, help='train registration network with loss terms dependent on generator')
#             parser.add_argument('--only_train_R', type=str2bool, nargs='?', const=True, default=False, help='only train registration network')
#             parser.add_argument('--only_train_G', type=str2bool, nargs='?', const=True, default=False, help='only train generator by CUT')
#             parser.add_argument('--train_G_pseudo', type=str2bool, nargs='?', const=True, default=False, help='only train generator by CUT')
            
            #new metric watcher
            parser.add_argument('--stn_cfg', type=str, default='A', help='Set the configuration used to build the STN.')
            parser.add_argument('--stn_type', type=str, default='pyramid', help='The type of STN to use. Currently supported are [affine, pyramid]')
            parser.add_argument('--stn_bilateral_alpha', type=float, default=0.0,
                            help='The bilateral filtering coefficient used in the the smoothness loss.'
                                 'This is relevant for unet stn only.')
            parser.add_argument('--stn_no_identity_init', action='store_true',
                            help='Whether to start the transformation from identity transformation or some random'
                                 'transformation. This is only relevant for unet stn (for affine the model'
                                 'doesn\'t converge).')
            parser.add_argument('--stn_multires_reg', type=int, default=1,
                            help='In multi-resolution smoothness, the regularization is applied on multiple resolution.'
                                 '(default : 1, means no multi-resolution)')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
#             self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            #only save G_A
            self.model_names = ['G_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        self.mini_bs = 1
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids)
        
        #new metrics watcher
        self.netR = models.stn.define_stn(self.opt, self.opt.stn_type)
        
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

###########################################################################################################################                
#     def set_input(self, input):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.

#         Parameters:
#             input (dict): include the data itself and its metadata information.

#         The option 'direction' can be used to swap domain A and domain B.
#         """
#         AtoB = self.opt.direction == 'AtoB'
#         self.real_A = input['A' if AtoB else 'B'].to(self.device)
#         self.real_B = input['B' if AtoB else 'A'].to(self.device)
#         self.image_paths = input['A_paths' if AtoB else 'B_paths']
#         x, y = input
#         self.real_A = x.to(self.device)
#         self.real_B = y.to(self.device)
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
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
            
        mini_dataset = torch.utils.data.TensorDataset(real_A_eq_patches, real_B_reg_patches)
        mini_dataloader = torch.utils.data.DataLoader(mini_dataset, batch_size=self.mini_bs, shuffle=False) 
        return mini_dataloader        
    
    def set_mini_input(self, input):
        x, y = input
        self.real_A_eq = x.to(self.device)
        self.real_B = y.to(self.device)
        
    def define_fold(self, data, dim = 512):
        self.dim = 512
        x, y = data[0][0], data[1][0]
        self.cx,h,w,self.cy  = y.size(1), x.size(2), x.size(3), y.size(1)
        padding = (int(np.ceil(h/self.dim)*self.dim-h)//2,int(np.ceil(w/self.dim)*self.dim-w)//2)
        self.unfold = torch.nn.Unfold(kernel_size=(self.dim,self.dim),padding=padding,stride=self.dim)
        self.fold = torch.nn.Fold(output_size=(h,w),kernel_size=(self.dim,self.dim),padding=padding,stride=self.dim)  
    
    
#     #investigate this part
#     def forward_stage2(self, fake_B=None):
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         warpped_images, self.deformation_field, reg_term = self.netR(self.real_A_eq, self.real_B, apply_on=[self.real_B, self.fake_A_eq])

#         self.stn_reg_term = reg_term
#         self.real_B_reg = warpped_images[0]
#         self.fake_A_reg_eq = warpped_images[1]
#         if fake_B != None:
#             self.fake_B = fake_B.to(self.device)  
            
# ############################################################################################################################
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A_eq)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

#         print("all shapes: ", self.real_A.shape, self.real_B.shape, self.fake_B.shape, self.rec_A.shape, self.fake_A.shape, self.rec_B.shape)
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_eq, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A_eq)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A_eq) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A_eq) * lambda_A
        content_loss_value = self.content_loss()
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
#         self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + content_loss_value
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B # without content loss
        self.loss_G.backward()

    def content_loss(self):

        L1_function = torch.nn.L1Loss()
        real_A_mean = torch.mean(self.real_A_eq,dim=1,keepdim=True)
        real_B_mean = torch.mean(self.real_B,dim=1,keepdim=True)
        fake_A_mean = torch.mean(self.fake_A,dim=1,keepdim=True)
        fake_B_mean = torch.mean(self.fake_B,dim=1,keepdim=True)

        real_A_normal = (real_A_mean - (self.opt.threshold_A))*100
        real_B_normal = (real_B_mean - (self.opt.threshold_B))*100

        fake_A_normal = (fake_A_mean - (self.opt.threshold_A))*100
        fake_B_normal = (fake_B_mean - (self.opt.threshold_B))*100

        real_A_sigmoid = torch.sigmoid(real_A_normal)
        real_B_sigmoid = 1 - torch.sigmoid(real_B_normal)
    
        fake_A_sigmoid = torch.sigmoid(fake_A_normal)
        fake_B_sigmoid = 1 - torch.sigmoid(fake_B_normal)
    
        content_loss_A = L1_function( real_A_sigmoid , fake_B_sigmoid )
        content_loss_B = L1_function( fake_A_sigmoid , real_B_sigmoid )

        content_loss_rate = 50*np.exp(-(self.opt.counter/self.opt.data_size))
        content_loss = (content_loss_A + content_loss_B)*content_loss_rate
        return content_loss


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.opt.counter += 1 # increment counter
        
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def save_images_FFOV(self, epoch, test_dataloader):
#         num_img = 2
#         img_dir = f'Results/{self.opt.name}/images/'        
#         if not os.path.exists(img_dir):
#             os.makedirs(img_dir)
#             print(f'Directory {img_dir} created')
#         else:
#             print(f'Directory {img_dir} already exists')  

#         _, ax = plt.subplots(num_img, 3, figsize=(40, 40))
#         [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Translated", "Target"])]
        
#         for i, data in enumerate(test_dataloader):
#             self.eval()
#             self.set_input(data)
#             self.define_fold(data)
#             if self.opt.LAB_space:
#                 target = lab_to_rgb(deprocess_lab(self.real_B.permute(0,2,3,1)))[0].detach().cpu().numpy()
#             else:
#                 target = (self.real_B.permute(0,2,3,1))[0].detach().cpu().numpy()
#             source = self.real_A_eq[0].permute([1,2,0]).detach().cpu().numpy() 
#             translated = self.patch_wise_predict(input_data=self.real_A_eq, stride_ratio=1).cpu().numpy()
#             translated = linear_normalize(translated) 

#             [ax[i,j].imshow(img) for j, img in enumerate([source, translated, target])]
#             [ax[i,j].axis("off") for j in range(3)]
        
#         plt.savefig(f'{img_dir}/epoch={epoch}.png')
#         plt.close()


        # new metric watcher
        num_img = 2
        metrics = np.zeros((12, num_img))   
        img_dir = f'Results/{self.opt.name}/images/'        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            print(f'Directory {img_dir} created')
        else:
            print(f'Directory {img_dir} already exists')  
            
#         if self.opt.only_train_G:
#             _, ax = plt.subplots(num_img, 3, figsize=(40, 40))
#             [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Translated", "Target"])]
#         if self.opt.only_train_R:
#             _, ax = plt.subplots(num_img, 4, figsize=(40, 40))
#             [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Target", "Registered", "Deformation field"])]
#         elif self.opt.train_G_pseudo:
#             _, ax = plt.subplots(num_img, 5, figsize=(40, 40))
#             [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Translated", "Target", "Source_pseudo", "Translated_pseudo"])]            
#         else:
        _, ax = plt.subplots(num_img, 3, figsize=(40, 40))
#         [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Translated", "Target", "Registered", "Deformation field"])]
        [ax[0,j].set_title(title) for j, title in enumerate(['Source', "Translated", "Target"])]
            
        for i, data in enumerate(test_dataloader):
            self.eval()
            self.set_input(data)
            
            self.define_fold(data)
            if self.opt.LAB_space:
                target = lab_to_rgb(deprocess_lab(self.real_B.permute(0,2,3,1)))[0].detach().cpu().numpy()
            else:
                target = (self.real_B.permute(0,2,3,1))[0].detach().cpu().numpy()
            source = self.real_A_eq[0].permute([1,2,0]).detach().cpu().numpy()
#             if self.opt.train_G_pseudo:    
#                 source_pseudo = self.fake_A_eq[0].permute([1,2,0]).detach().cpu().numpy()                   
            
#             if not self.opt.only_train_R:
            translated = self.patch_wise_predict(input_data=self.real_A_eq, stride_ratio=1).cpu().numpy()
            translated = linear_normalize(translated)   
            [ax[i,j].imshow(img) for j, img in enumerate([source, translated, target])]
            [ax[i,j].axis("off") for j in range(3)]
            metrics[7:11, i] = M.unpaired_fiber_metrics(translated, target)
            metrics[11, i] = M.vessel_area_diff(translated, target)

        plt.savefig(f'{img_dir}/epoch={epoch}.png')
        plt.close()
        return metrics.mean(1)
        
    def patch_wise_predict(self, input_data, patch_dim=512, bs=4, stride_ratio=4):
        dim = patch_dim
        stride = dim//stride_ratio
        X = input_data
        self.netG_A.eval()
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
            test_patches[bs*i:bs*(i+1),:,:,:] = self.netG_A(data[0].to(self.device)).permute([0,2,3,1]).detach().cpu()
        test = fold(test_patches.permute(3,1,2,0).reshape(cy,dim*dim,-1)).permute(1,2,3,0)
        weight = fold(weight_patches.permute(3,1,2,0).reshape(cy,dim*dim,-1)).permute(1,2,3,0)
        test = test / weight
        if self.opt.LAB_space:
            test = lab_to_rgb(deprocess_lab(test))
        return test.squeeze(0).detach()

#     # handle holding small nan values by adding eps
#     def patch_wise_predict(self, input_data, patch_dim=512, bs=4, stride_ratio=4):
#         dim = patch_dim
#         stride = dim//stride_ratio
#         X = input_data
#         self.netG_A.eval()
#         cx,h,w,cy  = X.shape[1], X.shape[2], X.shape[3],3
#         padding = (int(np.ceil(h/dim)*dim-h)//2,int(np.ceil(w/dim)*dim-w)//2)
#         unfold = torch.nn.Unfold(kernel_size=(dim,dim),padding=padding,stride=stride)
#         fold = torch.nn.Fold(output_size=(h,w),kernel_size=(dim,dim),padding=padding,stride=stride)
#         real_A_patches = unfold(X.permute(1,0,2,3)).reshape(cx,dim,dim,-1).permute(3,0,1,2)
#         test_dataset = torch.utils.data.TensorDataset(real_A_patches)
#         test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
#         test_patches = torch.Tensor(np.ndarray(shape=(real_A_patches.shape[0],real_A_patches.shape[2], 
#                                      real_A_patches.shape[3],3)))
#         weight_patches = torch.ones_like(test_patches)
#         eps = 1e-10
#         for i, data in enumerate(test_dataloader):
#             output = self.netG_A(data[0].to(self.device))
#             output[torch.isnan(output)] = eps
#             test_patches[bs*i:bs*(i+1),:,:,:] = output.permute([0,2,3,1]).detach().cpu()
#         test = fold(test_patches.permute(3,1,2,0).reshape(cy,dim*dim,-1)).permute(1,2,3,0)
#         weight = fold(weight_patches.permute(3,1,2,0).reshape(cy,dim*dim,-1)).permute(1,2,3,0)
#         weight[weight < eps] = eps
#         test = test / weight
#         if self.opt.LAB_space:
#             test = lab_to_rgb(deprocess_lab(test))
#         return test.squeeze(0).detach()

