import torch

from .affine_stn import AffineSTN
from .unet_stn import PyramidUnetSTN

sampling_align_corners = False
sampling_mode = 'bilinear'


def modify_commandline_options(parser, is_train=True):
    parser.add_argument('--stn_cfg', type=str, default='A', help='Set the configuration used to build the STN.')
    parser.add_argument('--stn_type', type=str, default='pyramid',
                        help='The type of STN to use. Currently supported are [affine, pyramid]')
    if is_train:
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


# def define_stn(opt, stn_type='pyramid'):
#     """Create and return an STN model with the relevant configuration."""
# #     def wrap_multigpu(stn_module, opt):
# #         if len(opt.gpu_ids) > 0:
# #             assert (torch.cuda.is_available())
# #             stn_module.to(opt.gpu_ids[0])
# #             stn_module = torch.nn.DataParallel(stn_module, opt.gpu_ids)  # multi-GPUs
# #         return stn_module

#     nc_a = opt.input_nc if opt.direction == 'AtoB' else opt.output_nc
#     nc_b = opt.output_nc if opt.direction == 'AtoB' else opt.input_nc
#     height = opt.img_height
#     width = opt.img_width
#     cfg = opt.stn_cfg
#     stn = None
#     if stn_type == 'affine':
#         stn = AffineSTN(nc_a, nc_b, height, width, cfg, opt.init_type)
#     if stn_type == 'pyramid':
#         stn = PyramidUnetSTN(nc_a, nc_b, height, width, cfg, opt.init_type, opt.stn_bilateral_alpha,
#                              (not opt.stn_no_identity_init), opt.stn_multires_reg)
# #     return wrap_multigpu(stn, opt)
#     return stn

def define_stn(opt, stn_type='pyramid'):
    """Create and return an STN model with the relevant configuration."""
#     def wrap_multigpu(stn_module, opt):
#         if len(opt.gpu_ids) > 0:
#             assert (torch.cuda.is_available())
#             stn_module.to(opt.gpu_ids[0])
#             stn_module = torch.nn.DataParallel(stn_module, opt.gpu_ids)  # multi-GPUs
#         return stn_module

    nc_a = opt.input_nc if opt.direction == 'AtoB' else opt.output_nc
    nc_b = opt.output_nc if opt.direction == 'AtoB' else opt.input_nc
    cfg = opt.stn_cfg
    stn = None
    if stn_type == 'affine':
        stn = AffineSTN(nc_a, nc_b, height, width, cfg, opt.init_type)
    if stn_type == 'pyramid':
        stn = PyramidUnetSTN(nc_a, nc_b, cfg, opt.init_type, opt.stn_bilateral_alpha,
                             (not opt.stn_no_identity_init), opt.stn_multires_reg)
#     return wrap_multigpu(stn, opt)
    return stn