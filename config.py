from easydict import EasyDict as edict
from md_classification3d.utils.vcls_helpers import FixedNormalizer, AdaptiveNormalizer
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# model transfer learning method
__C.general.transfer = False

# image-label pair list
# training csv file, head format
# single modality [image_path, class, x, y, z, width, height, depth]
# multi modality [image_path, image_path1, ..... , class, x, y, z, width, height, depth]
__C.general.im_classification_list = 'new.csv'

# test after save pytorch model
# testing train dataset
__C.general.is_train = True

__C.general.is_valid = None
__C.general.im_valid_list = None

# testing validation dataset
__C.general.is_test = True
__C.general.im_test_list = "test_ards.csv"

# the output of training models and logs
__C.general.save_dir = '/data/yan/ards2026_2/ResNetCAM0901_256'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training
__C.general.num_gpus = 1

# random seed used in training (debugging purpose), -1 for random
__C.general.seed = -1

##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channels
__C.dataset.input_channel = 1

# the number of classes
__C.dataset.num_classes = num_classes = 2

# index for label and label name
__C.dataset.label_name_index = {'0': 0, '1': 1}

# crop intensity normalizers (to [-1, 1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [FixedNormalizer(mean=-600, stddev=750, clip=True)]
# __C.dataset.crop_normalizers = [FixedNormalizer(mean=40, stddev=175, clip=True)]
                                # FixedNormalizer(mean=0.5, stddev=0.5, clip=True)]

# each label has the same number in one batch
__C.dataset.equal_sample = True

# each label has the custom sample frequency number in one batch
# __C.dataset.sample_frequency = np.ones(num_classes) / num_classes
__C.dataset.sample_frequency = None

# fixed_length:
# fixed_box:
# fixed_box_nn:
__C.dataset.sample_method = "fixed_length"

# input voxel size (w, h, d)
__C.dataset.crop_size = [256, 256, 128]

# random flip input crop
__C.dataset.random_flip = True
__C.dataset.flip_config = {'random_flip': True, 'flip_x_prob': 0, 'flip_y_prob': 0, 'flip_z_prob': 0.5}

# random rotate input crop in degrees
__C.dataset.rotate_config = {'rot_prob': 0.5, 'rot_angle_degree': 360, 'rot_axis': [[0, 0, 1]]}

# random scale ratio of input crop
__C.dataset.scale_config = {'scale_prob': 0.5, 'scale_min_ratio': 0.75, 'scale_max_ratio': 1.25,
                            'scale_isotropic': True}

# random mix-up of input crop
__C.dataset.mixup_config = {'mixup_prob': 0.5, 'alpha': 0.1}

# parameter for fixed_length
# box_center_random (unit: mm)
__C.dataset.spacing = [1, 1, 3]
__C.dataset.box_center_random = [24, 24, 24]

# parameter for fixed_box or fixed_box_nn
# box_percent_padding, the percent of nodule bounding box padding on each side
# box_center_random_percent, random range of center point
__C.dataset.box_percent_padding = 0.5
__C.dataset.box_center_random_percent = 0.5
__C.dataset.interpolation = ['linear']

####################################
# training loss
####################################

__C.loss = {}

####################################
# loss
# Ap
# Focal  alpha_matrix, gamma
# LabelSmooth
####################################
__C.loss.name = 'Focal'
__C.loss.params = {'alpha_matrix': np.array([[1, 0], [0, 1]]),
                   'gamma': 0}

# __C.loss.params = {'smoothing': 0.1}
__C.loss.cam_loss_weight = 0.3
__C.loss.cam_dice_beta = 1
__C.loss.camname = 'Dice'

#####################################
# net
# the network name BasicNet, ResidualNet, DenseNet, BResidualNet, BasicNetP, ResidualNetUseMask
#####################################
__C.net = {}
__C.net.name = 'ResidualNetUseMask'
__C.net.cam_w = 100
__C.net.cam_sigma = 0.4

######################################
# training parameters
######################################
__C.train = {}
__C.train.epochs = 2001
__C.train.batchsize = 8
__C.train.num_threads = 64
__C.train.lr = 0.0001

####################################
# CosineAnnealing 参数 T_max,eta_min,last_epoch
# Step            参数 step_size, gamma, last_epoch
# MultiStep       参数 milestones, gamma, last_epoch
# Exponential     参数 gamma, last_epoch
# 1�?last_epoch如果没有设置或者�ye��置为-1，last_epoch将被设置为__C.general.resume_epoch
# 2�?方法还有很多，自己pytorch查询
####################################
__C.train.lr_scheduler = {}
__C.train.lr_scheduler.name = "Step"
__C.train.lr_scheduler.params = {"step_size": 100, "gamma": 0.8, "last_epoch": -1}

# Warmup method
# linear: UntunedLinearWarmup
# exponential: UntunedExponentialWarmup
# radam: RAdamWarmup
# none: LinearWarmup
# GradualWarmup: GradualWarmupyes
__C.train.lr_scheduler.warmup = 'GradualWarmup'
__C.train.lr_scheduler.warmup_multiplier = 1.0
__C.train.lr_scheduler.warmup_total_epoch = 100

####################################
# 方法 Adam           参数 betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False
# 方法 SGD            参数 momentum=0, dampening=0, weight_decay=0, nesterov=False
####################################
__C.train.optimizer = {}
__C.train.optimizer.name = "Adam"
__C.train.optimizer.params = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False}
# __C.train.optimizer.params = {"momentum":0.05, "weight_decay":1e-8}

# the number of batches to update loss curve
__C.train.plot_snapshot = 100

# the number of batches to save model
__C.train.save_epochs = 5
__C.train.save_inputs = True

