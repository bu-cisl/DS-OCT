import skimage as ski
from skimage.filters import thresholding, rank, median, apply_hysteresis_threshold, farid_v, farid_h
from skimage.morphology import binary_erosion, binary_dilation,binary_closing, disk, binary_opening, diameter_opening, reconstruction, square, remove_small_objects, remove_small_holes
from skimage.measure import regionprops_table
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from frangi_sc import frangi
import pandas as pd
import scipy
import numpy as np
import cv2 as cv

def linear_normalize(tmp):
    return (tmp - tmp.min())/(tmp.max() - tmp.min())

def hy_th(img, low, high):
    return apply_hysteresis_threshold(img, low, high)

def find_layer_mask(Gallyas_img):
    img = Gallyas_img.mean(axis=2) if len(Gallyas_img.shape) == 3 else Gallyas_img
    mask_smp = remove_small_holes(remove_small_objects(img < thresholding.threshold_triangle(img),256), 512)
    mask_456 = distance_transform_edt(mask_smp)>75 ## theoretical thickness of 1/2/3 layer is 1300 um ~ 108 pixels
    if len(Gallyas_img.shape) == 3:
        mask_GM = binary_dilation(fill_white_holes_with_black(img > thresholding.threshold_minimum(img)),square(8))
        mask_WM = True ^ mask_GM
        I = cv.cvtColor(Gallyas_img, cv.COLOR_RGB2LAB).astype(np.float32)
        L, A, B = cv.split(I)
        L_mean, A_mean, B_mean = L[mask_WM].mean(), A[mask_WM].mean(), B[mask_WM].mean()
        img = (L-L_mean)**2+(A-A_mean)**2+(B-B_mean)**2
    else:
        mask_GM = binary_dilation(fill_white_holes_with_black(img > thresholding.threshold_otsu(img)),square(8))
        mask_WM = True ^ mask_GM
        img_mean = img[mask_WM].mean()
        img = (img-img_mean)**2
    th_brown = thresholding.threshold_mean(img)
    mask_brown = img < th_brown
    mask_brown = remove_small_holes(remove_small_objects(mask_brown,256),512)
    return mask_GM*mask_456*mask_brown, mask_smp

def fill_white_holes_with_black(mask):
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = mask.min()
    rec = reconstruction(seed, mask, method='dilation')
    return rec

def fill_holes(mask):
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = mask.max()
    rec = reconstruction(seed, mask, method='erosion')
    return rec

def flatten_image_to_valid_1Darray(mask, val):
    val = val.mean(axis=2,keepdims=False) if len(val.shape) ==3 else val
    flt_mask, flt_val = mask.ravel().astype(np.bool), val.ravel()
    flt_valid_val = flt_val[flt_mask]
    return flt_valid_val

def IOU(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    overlap = mask1*mask2
    union = mask1 + mask2
    return overlap.sum()/float(union.sum())

def angle(ori):
    return np.arctan2(abs(ori[1]), abs(ori[0]))

def edg_angle(img):
    return np.arctan2(abs(farid_h(img)), abs(farid_v(img)))

def fr(img):
    filt_max, ori = frangi(linear_normalize(img), sigmas=np.arange(0.4, 1.0, 0.1), beta=0.25,gamma=15)
    return linear_normalize(filt_max), ori

def extract_fiber(Gallyas_img, th_fr=0.005):
    layer_mask, mask_smp = find_layer_mask(Gallyas_img)
    dist_map = distance_transform_edt(mask_smp)
    if len(Gallyas_img.shape) == 3:
        fr_filt, ori = fr(Gallyas_img.mean(axis=2))
    else:
        fr_filt, ori  = fr(Gallyas_img)
    fiber_mask = hy_th(fr_filt, 0.001, th_fr)*layer_mask
    prob_map = np.exp(-(angle(ori) - edg_angle(dist_map)) ** 2)
    fiber_mask = (prob_map > 0.6) * fiber_mask
    fiber_mask_denoised = diameter_opening(fiber_mask, 10)
    fiber_mask_denoised = fiber_mask_denoised[:,:,None] if len(Gallyas_img.shape) == 3 else fiber_mask_denoised
    return fiber_mask_denoised, layer_mask

def compute_fiber_df(fiber_mask_denoised):
    fiber_mask_denoised = fiber_mask_denoised.squeeze()
    '''compute fiber mask metrics, could include orientation, area[200:500, 1800:2400], axis_minor_length'''
    lb_img = ski.measure.label(fiber_mask_denoised)
    df = pd.DataFrame(regionprops_table(lb_img, properties=['label','axis_major_length','area']))
    df['diameter'] = df['area']/(df['axis_major_length']+1e-8)
    df = df[(df['axis_major_length'] < 800 / 12.0) * (df['diameter'] < 50 / 12.0)]
    return df

def extract_vessel_WM(Gallyas_img):
    if len(Gallyas_img.shape) == 3:
        img = Gallyas_img.mean(axis=2)
        th_WM = thresholding.threshold_minimum(img)
    else:
        img = Gallyas_img
        th_WM = thresholding.threshold_otsu(img)
    mask_WM = binary_dilation(binary_erosion(fill_holes(img < th_WM), disk(10)), disk(3))
    vessels = remove_small_objects(binary_opening(mask_WM * (img > th_WM), disk(1)), 10)
    return vessels

'''quantitative metrics include: 
pixel-wise: MSE, PCC, SSIM, Color diff, fiber PCC, layer IOU, vessel IOU
population-level: JS of fiber length, JS of fiber diameter, fiber area diff, fiber number diff, vessel area diff'''
def MSE(img1, img2):
    return mean_squared_error(img1, img2)

def PCC(img1, img2):
    return np.corrcoef(img1.ravel(), img2.ravel())[0, 1]

def SSIM(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2)

def CD(img1, img2):
    I = cv.cvtColor(img1, cv.COLOR_RGB2LAB).astype(np.float32)
    L1, A1, B1 = cv.split(I)
    I = cv.cvtColor(img2, cv.COLOR_RGB2LAB).astype(np.float32)
    L2, A2, B2 = cv.split(I)
    return np.sqrt((L1 - L2)**2 + (A1 - A2)**2 + (B1 - B2)**2).mean()

'''fiber_PCC and layer_IOU'''
def fiber_PCC_and_layer_IOU(img1, img2):
    layer_mask1, _ = find_layer_mask(img1)
    layer_mask2, _ = find_layer_mask(img2)
    common_mask = (layer_mask1 + layer_mask2)>0
    flt_valid_val1 = flatten_image_to_valid_1Darray(common_mask, img1)
    flt_valid_val2 = flatten_image_to_valid_1Darray(common_mask, img2)
    fiber_PCC = np.corrcoef(flt_valid_val1, flt_valid_val2)[0, 1]
    layer_IOU = IOU(layer_mask1, layer_mask2)
    return fiber_PCC, layer_IOU

def vessel_IOU(img1, img2):
    vessel1 = extract_vessel_WM(img1)
    vessel2 = extract_vessel_WM(img2)
    return IOU(vessel1, vessel2)

'''JS_length, JS_diameter'''
def compute_JS_fiber_length_and_diameter(df1, df2):
    pk, _ = np.histogram(df1['axis_major_length'] * 12, bins=256, range=(0, 800), density=True)
    qk, _ = np.histogram(df2['axis_major_length'] * 12, bins=256, range=(0, 800), density=True)
    JS_length = scipy.spatial.distance.jensenshannon(pk, qk)
    pk, _ = np.histogram(df1['diameter'] * 12, bins=256, range=(0, 50), density=True)
    qk, _ = np.histogram(df2['diameter'] * 12, bins=256, range=(0, 50), density=True)
    JS_diameter = scipy.spatial.distance.jensenshannon(pk, qk)
    return JS_length, JS_diameter

'''fiber_area_diff, fiber_num_diff'''
def compute_diff_fiber_area_and_number(df, df_gt):
    area_diff = abs(df['area'].sum() - df_gt['area'].sum()) / df_gt['area'].sum()
    num_diff = abs(len(df['label']) - len(df_gt['label'])) / len(df_gt['label'])
    return area_diff, num_diff

def vessel_area_diff(vessel1, vessel_gt):
    return abs(vessel1.sum() - vessel_gt.sum()) / vessel_gt.sum()


