import numpy as np
from scipy import ndimage
from random import randrange, shuffle
from torch.utils.data import Dataset

def data_aug_preprocessing_HR(X, Y, dim=512, num=16, unaligned=False):
    # for splitting large image into 512x512 FOV
    ## X, Y dimension order as (N,C,H,W), need to permute first ##
    X, Y = np.transpose(X, [0,2,3,1]).squeeze(0), np.transpose(Y, [0,2,3,1]).squeeze(0)
    cx, cy = X.shape[2], Y.shape[2]
    
    if not unaligned:
        data_augmented = np.ndarray(shape=(6*num,dim,dim,cx+cy))
        Data = np.concatenate([X, Y], axis=-1)
        for i in range(num):
            tmp = np.ndarray(shape=(6,dim,dim,cx+cy))
            x1 = randrange(0, Data.shape[0] - dim)
            y1 = randrange(0, Data.shape[1] - dim)
            tmp[0,:] = Data[x1:x1+dim, y1:y1+dim,:]
            tmp[1,:] = np.fliplr(tmp[0,:])
            tmp[2,:] = np.flipud(tmp[0,:])
            tmp[3,:] = ndimage.rotate(tmp[0,:], 90)
            tmp[4,:] = ndimage.rotate(tmp[0,:], 180)
            tmp[5,:] = ndimage.rotate(tmp[0,:], 270)
            data_augmented[6*i:6*i+6,:] = tmp

        shuffle(data_augmented)
        x_augmented = data_augmented[:,:,:,0:cx].astype('float32')
        y_augmented = data_augmented[:,:,:,cx:].astype('float32')
        x_augmented = np.transpose(x_augmented, [0,3,1,2])
        y_augmented = np.transpose(y_augmented, [0,3,1,2])
    else:
        x_augmented = np.ndarray(shape=(6*num,dim,dim,cx))
        y_augmented = np.ndarray(shape=(6*num,dim,dim,cy))
        for i in range(num):
            tmp = np.ndarray(shape=(6,dim,dim,cx))
            # random cropping
            x1 = randrange(0, X.shape[0] - dim)
            y1 = randrange(0, X.shape[1] - dim)
            tmp[0,:] = X[x1:x1+dim, y1:y1+dim,:]
            # simple data augmentation
            tmp[1,:] = np.fliplr(tmp[0,:])
            tmp[2,:] = np.flipud(tmp[0,:])
            tmp[3,:] = ndimage.rotate(tmp[0,:], 90)
            tmp[4,:] = ndimage.rotate(tmp[0,:], 180)
            tmp[5,:] = ndimage.rotate(tmp[0,:], 270)
            x_augmented[6*i:6*i+6,:] = tmp
        for i in range(num):
            tmp = np.ndarray(shape=(6,dim,dim,cy))
            x1 = randrange(0, Y.shape[0] - dim)
            y1 = randrange(0, Y.shape[1] - dim)
            tmp[0,:] = Y[x1:x1+dim, y1:y1+dim,:]
            tmp[1,:] = np.fliplr(tmp[0,:])
            tmp[2,:] = np.flipud(tmp[0,:])
            tmp[3,:] = ndimage.rotate(tmp[0,:], 90)
            tmp[4,:] = ndimage.rotate(tmp[0,:], 180)
            tmp[5,:] = ndimage.rotate(tmp[0,:], 270)
            y_augmented[6*i:6*i+6,:] = tmp
            
        shuffle(x_augmented),shuffle(y_augmented)
        # converting [N,H,W,C] to [N,C,H,W]
        x_augmented = np.transpose(x_augmented.astype('float32'), [0,3,1,2])
        y_augmented = np.transpose(y_augmented.astype('float32'), [0,3,1,2])
        
    return x_augmented, y_augmented

import torch
def patch_wise_predict(model, input_data, patch_dim=512, bs=4, stride_ratio=4):
    dim = patch_dim
    stride = dim//stride_ratio
    X = input_data
    model.netG.eval()
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
        test_patches[bs*i:bs*(i+1),:,:,:] = model.netG(data[0].to(model.device)).permute([0,2,3,1]).detach().cpu()
    test = fold(test_patches.permute(3,1,2,0).reshape(cy,dim*dim,-1)).permute(1,2,3,0).squeeze(0)
    weight = fold(weight_patches.permute(3,1,2,0).reshape(cy,dim*dim,-1)).permute(1,2,3,0).squeeze(0)
    test = test / weight
    return test

def VariedSizedImagesCollate(batch):
    data_x = [item[0] for item in batch]
    data_y = [item[1] for item in batch]
    ## should only work when batch_size==1
    return [data_x, data_y]

def LoadingImgs_to_ListOfListOfTensors(data_folder, list_img_index):
    list_of_list_of_tensors = []
    for i in list_img_index:
        X = 1 - np.load(data_folder + 'x_mus_'+str(i)+'.npy').astype('float32')[np.newaxis,:,:,np.newaxis]
        Y = np.load(data_folder + 'y_gallyas_'+str(i)+'.npy').astype('float32')[np.newaxis,:,:,:]
        list_of_list_of_tensors.append([torch.Tensor(np.transpose(X, [0,3,1,2])),torch.Tensor(np.transpose(Y, [0,3,1,2]))])
        
    return list_of_list_of_tensors

class VariedSizedImagesDataset(Dataset):
    def __init__(self, ListOfListOfTensors):
        self.data = ListOfListOfTensors

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

import cv2
import torch
import numpy as np
from scipy import misc
from PIL import Image

def preprocess_lab(lab):
		L_chan, a_chan, b_chan =torch.unbind(lab,dim=3)
		# L_chan: black and white with input range [0, 100]
		# a_chan/b_chan: color channels with input range ~[-110, 110], not exact
		# [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
		return torch.stack([L_chan / 100, a_chan / 220.0 + 0.5, b_chan / 220.0 + 0.5], dim = 3)


def deprocess_lab(LAB_chan):
		#TODO This is axis=3 instead of axis=2 when deprocessing batch of images 
			   # ( we process individual images but deprocess batches)
		#return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
		L_chan, a_chan, b_chan =torch.unbind(LAB_chan,dim=3)
		return torch.stack([L_chan* 100.0, (a_chan - 0.5)* 220.0, (b_chan - 0.5) * 220.0], dim=3)


def rgb_to_lab(srgb):

	srgb_pixels = torch.reshape(srgb, [-1, 3]).cpu()

	linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor)
	exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor)
	rgb_pixels = ((srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask).type(torch.FloatTensor)
	
	rgb_to_xyz = torch.tensor([
				#    X        Y          Z
				[0.412453, 0.212671, 0.019334], # R
				[0.357580, 0.715160, 0.119193], # G
				[0.180423, 0.072169, 0.950227], # B
			]).type(torch.FloatTensor)
	
	xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
	

	# XYZ to Lab
	xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor))

	epsilon = 6.0/29.0

	linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor)

	exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor)

	fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
	# convert to lab
	fxfyfz_to_lab = torch.tensor([
		#  l       a       b
		[  0.0,  500.0,    0.0], # fx
		[116.0, -500.0,  200.0], # fy
		[  0.0,    0.0, -200.0], # fz
	]).type(torch.FloatTensor)
	lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor)
	#return tf.reshape(lab_pixels, tf.shape(srgb))
	return torch.reshape(lab_pixels, srgb.shape)

def lab_to_rgb(lab):
		lab_pixels = torch.reshape(lab, [-1, 3]).cpu()
		# convert to fxfyfz
		lab_to_fxfyfz = torch.tensor([
			#   fx      fy        fz
			[1/116.0, 1/116.0,  1/116.0], # l
			[1/500.0,     0.0,      0.0], # a
			[    0.0,     0.0, -1/200.0], # b
		]).type(torch.FloatTensor)
		fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor), lab_to_fxfyfz)

		# convert to xyz
		epsilon = 6.0/29.0
		linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor)
		exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor)


		xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

		# denormalize for D65 white point
		xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor))


		xyz_to_rgb = torch.tensor([
			#     r           g          b
			[ 3.2404542, -0.9692660,  0.0556434], # x
			[-1.5371385,  1.8760108, -0.2040259], # y
			[-0.4985314,  0.0415560,  1.0572252], # z
		]).type(torch.FloatTensor)

		rgb_pixels =  torch.mm(xyz_pixels, xyz_to_rgb)
		# avoid a slightly negative number messing up the conversion
		#clip
		rgb_pixels[rgb_pixels > 1] = 1
		rgb_pixels[rgb_pixels < 0] = 0

		linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor)
		exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor)
		srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask
	
		return torch.reshape(srgb_pixels, lab.shape)
    
