import numpy as np
from util.data_preparation import VariedSizedImagesCollate, LoadingImgs_to_ListOfListOfTensors, VariedSizedImagesDataset, data_aug_preprocessing_HR, patch_wise_predict
import time
import torch
import torchvision.transforms as T
from options.train_options import TrainOptions
from models import create_model
def linear_normalize(tmp):
    return (tmp - tmp.min())/(tmp.max() - tmp.min())
import matplotlib.pyplot as plt
import os
from PIL import Image
import imageio

'''load data'''
# data_folder = '../Data/Data_1212_22/'
data_folder = '../Data/Data_0309_23/'

list_of_list_of_tensors = []
num_imgs = 35
for i in range(1, num_imgs+1):
    # X = 1 - np.array(Image.open(data_folder + 'OCT_'+str(i)+'.tif')).astype('float32')[np.newaxis, np.newaxis, :, :]/255.0
    # Y = np.array(Image.open(data_folder + 'GS_stain_normalized_'+str(i)+'.tif')).astype('float32')[np.newaxis,:,:,:]/255.0
    X = 1.0 - np.load(data_folder+'x_mus_'+str(i)+'.npy').astype('float32')[np.newaxis, np.newaxis, :, :]
    Y = np.load(data_folder+'y_gallyas_'+str(i)+'.npy').astype('float32')[np.newaxis,:,:,:]
    list_of_list_of_tensors.append([torch.Tensor(X), torch.Tensor(np.transpose(Y, [0,3,1,2]))])
dataset = VariedSizedImagesDataset(list_of_list_of_tensors)
## batch_size can only be 1
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1, collate_fn=VariedSizedImagesCollate, shuffle=False,
                                         num_workers=0, drop_last=True)

'''CycleGAN model weights'''
model = '--model cycle_gan'
opt = TrainOptions(model).parse()   # get training options
model = create_model(opt)
G_weights_name = 'CycleGAN/465_net_G_A'
model.netG_A.load_state_dict(torch.load('checkpoints/'+G_weights_name+'.pth'))

# model = '--model cycle_gan --netG unet_256 --lambda_identity 0'
# opt = TrainOptions(model).parse()   # get training options
# model = create_model(opt)      # create a model given opt.model and other options
# opt.model = 'cycle_gan'
# opt.netG = 'unet_256'
# model = create_model(opt)
# G_weights_name = 'batch_A1_B3_without_compute_thAB/100_net_G_A'
# state_dict = torch.load('checkpoints/'+G_weights_name+'.pth')
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove module.
#     new_state_dict[name] = v
# model.netG_A.load_state_dict(new_state_dict)
# model.netG_A.to(model.device)


'''create model and load model weights'''
# model = '--model cutreg_twostage'
# opt = TrainOptions(model).parse()   # get training options
# opt.input_nc = 3
# model = create_model(opt)      # create a model given opt.model and other options

'''CUT model weights'''
# G_weights_name = 'adam_lin_GAN1_NCEX20_NCEY10_lr0.0003_apr26_CUT/190_net_G'

'''FastCUT model weights'''
# G_weights_name = 'CUT_9spl_4x_GAN1_NCE20_3Channel/280_net_G'

'''Pseudo_supervised model weights'''
# G_weights_name = 'PseudoSupervised_9spl_4x_MAEpseudo10_3channel/270_net_G'

'''Pseudo + CUT + MAE_reg model weights'''
# G_weights_name = 'CUT+REG+Pseudo_9spl_4x_NCE20_REG100_TV600_MAE1_MAEpseudo10_3channel/260_net_G'

# '''Pseudo + GAN model weights'''
# # weights_name = 'CUT+REG+Pseudo_18spl_4x_NCE0_REG100_TV600_MAE0_MAEpseudo10/180_net_G'
# '''Pseudo + MAE_reg + GAN model weights'''
# # weights_name = 'CUT+REG+Pseudo_18spl_4x_NCE0_REG100_TV600_MAE1_MAEpseudo10/200_net_G'
# '''Pseudo + CUT (GAN+contrastiveNCE) model weights'''
# # weights_name = 'CUT+REG+Pseudo_18spl_4x_NCE10_REG100_TV600_MAE0_MAEpseudo10/180_net_G'
# '''Pseudo + CUT + LAB_loss model weights'''
# # G_weights_name = 'CUT+REG+Pseudo_9spl_4x_NCE20_REG100_TV600_LAB01_MAEpseudo10/290_net_G'
# '''Pseudo + CUT + MAE_reg + LAB_loss model weights'''
# # G_weights_name = 'CUT+REG+Pseudo+LAB_9spl_4x_NCE20_REG100_TV600_MAE1_LAB01_MAEpseudo10_3channel/335_net_G'
# model.netG.load_state_dict(torch.load('checkpoints/'+G_weights_name+'.pth'))

test_list_numpy = []
gt_list_numpy = []

img_dir = '../Data/Test_results/' + G_weights_name + '/processed/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
    print(f'Directory {img_dir} created')
else:
    print(f'Directory {img_dir} already exists')

for i, data in enumerate(dataloader):
    model.set_input(data)
    fake_B = model.patch_wise_predict(input_data=model.real_A_eq, bs=2, stride_ratio=4).cpu().numpy()
    imageio.imwrite(img_dir + 'translated_' + str(i + 1) + '.tif', (fake_B*255).astype(np.uint8))
    np.save(img_dir + 'translated_' + str(i + 1) + '.npy', fake_B)


# '''generate registration results from netR'''
# R_weights_name = 'REG_35spl_4x_MAE_REG100TV600_ResUnet_3channel/500_net_R'
# model.netR.load_state_dict(torch.load('checkpoints/'+R_weights_name+'.pth'))
# test_R_target_list_numpy = []
# test_R_moving_OD_list_numpy = []
# test_R_OD_list_numpy = []
# test_R_GS_list_numpy = []
# img_dir = '../Data/Test_results/' + R_weights_name + '/'
# if not os.path.exists(img_dir):
#     os.makedirs(img_dir)
#     print(f'Directory {img_dir} created')
# else:
#     print(f'Directory {img_dir} already exists')
# for i, data in enumerate(dataloader):
#     model.set_input(data)
#     model.forward_stage2()
#     test_R_target_list_numpy.append(1.0 - model.real_A_eq.permute(0,2,3,1)[:,:,:,0].squeeze().detach().cpu().numpy())
#     test_R_moving_OD_list_numpy.append(1.0 - model.fake_A_eq.permute(0,2,3,1)[:,:,:,0].squeeze().detach().cpu().numpy())
#     model.forward_stage2()
#     test_R_OD_list_numpy.append(1.0 - model.fake_A_reg_eq.permute(0,2,3,1).squeeze().detach().cpu().numpy())
#     test_R_GS_list_numpy.append(model.real_B_reg.permute(0,2,3,1).squeeze().detach().cpu().numpy())
#
# for i in range(len(test_R_target_list_numpy)):
#     imageio.imwrite(img_dir+'OCT_eq'+str(i + 1)+'.tif', (test_R_target_list_numpy[i]*255).astype(np.uint8))
#     imageio.imwrite(img_dir+'moving_OD'+str(i + 1)+'.tif', (test_R_moving_OD_list_numpy[i]*255).astype(np.uint8))
#     imageio.imwrite(img_dir+'moved_OD'+str(i + 1)+'.tif', (test_R_OD_list_numpy[i]*255).astype(np.uint8))
#     imageio.imwrite(img_dir+'moved_GS'+str(i + 1)+'.tif', (test_R_GS_list_numpy[i]*255).astype(np.uint8))
