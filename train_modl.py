#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:00:26 2020

@author: marius
"""
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '.')

import torch, os, glob, h5py
import numpy as np
from tqdm import tqdm
from dotmap import DotMap

from aux_brain_generators import MCFullFastMRI, crop
from aux_unrolled_cplx_modl import MoDLDoubleUnroll
from aux_losses import SSIMLoss, MCLoss
from aux_utils import ifft

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 6})
plt.ioff(); plt.close('all')

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Fix seed
global_seed = 1999
torch.manual_seed(global_seed)
np.random.seed(global_seed)
# Enable cuDNN kernel selection
torch.backends.cudnn.benchmark = True

## Training files
core_dir    = '/your/train/folder/here'
train_files = sorted(glob.glob(core_dir + '/*T2*.h5'))
# Filter all train files to only retain T2 @ 3T scans
full_rows    = 768
target_lines = 384
filtered_idx = []
for idx, file in tqdm(enumerate(train_files)):
    with h5py.File(file, 'r') as contents:
        # Just check the size
        (num_slices, num_coils, h, w) = contents['kspace'].shape
        # Skip if size is not 768 x >384
        if h != full_rows or w < target_lines:
            pass
        else:
            filtered_idx.append(idx)
filtered_train_files = [train_files[idx] for idx in filtered_idx]
print('Sub-selected %d training files!' % len(train_files))

# MoDL also requires maps
maps_dir = '/your/train/maps/here'
train_maps, train_files = [], []
for idx, file in enumerate(filtered_train_files):
    if os.path.exists(os.path.join(maps_dir, os.path.basename(file))):
        train_files.append(os.path.join(core_dir, os.path.basename(file)))
        train_maps.append(os.path.join(maps_dir, os.path.basename(file)))

# Data config
num_slices = 'all'
# Config
hparams         = DotMap()
hparams.mode    = 'MoDL'
hparams.logging = False

# ResNet parameters
hparams.img_channels = 64
hparams.img_blocks   = 4 # '0' means bypass
# Data
hparams.downsample = [4, 8] # Mixture
hparams.use_acs    = True
hparams.acs_lines  = 0 # A fixed number, unused if above is 'True'
# Model
hparams.use_img_net        = True
hparams.use_map_net        = True
hparams.map_init           = 'espirit' if hparams.mode == 'MoDL' else 'estimated'
hparams.img_init           = 'estimated'
hparams.loss_space         = 'mvue'
hparams.map_norm           = False
hparams.img_net_arch       = 'ResNet'
hparams.mps_kernel_shape   = [15, 9]
hparams.l2lam_init         = 0.01
hparams.l2lam_train        = True
hparams.crop_rss           = True
hparams.num_unrolls        = 6 # Starting value
hparams.block1_max_iter    = 0 if hparams.mode == 'MoDL' else 6
hparams.block2_max_iter    = 6
hparams.cg_eps             = 1e-6
hparams.verbose            = False
# Static training parameters
hparams.lr           = 2e-4 # Finetune if desired
hparams.num_epochs   = 15
hparams.step_size    = 10
hparams.decay_gamma  = 0.5
hparams.grad_clip    = 1.
hparams.start_epoch  = 0 # Warm start if desired
hparams.batch_size   = 1 # Unsupported w/ dynamic samples

# Global directory
global_dir = 'models/%s' % hparams.loss_space
if not os.path.exists(global_dir):
    os.makedirs(global_dir)

# Datasets
train_dataset = MCFullFastMRI(train_files, num_slices,
                              downsample=hparams.downsample,
                              use_acs=hparams.use_acs, 
                              acs_lines=hparams.acs_lines,
                              mps_kernel_shape=hparams.mps_kernel_shape,
                              maps=train_maps)
train_loader  = DataLoader(train_dataset, batch_size=hparams.batch_size, 
                           shuffle=True, num_workers=12, drop_last=True)

# Get a sample-specific model
model = MoDLDoubleUnroll(hparams)
model = model.cuda()
num_params = np.sum([np.prod(p.shape) for p in model.parameters()])
print('Model has %d parameters.' % num_params)
# Switch to train
model.train()

# Criterions
ssim           = SSIMLoss().cuda()
multicoil_loss = MCLoss().cuda()
pixel_loss     = torch.nn.MSELoss(reduction='sum')


# Get a local optimizer and scheduler
optimizer = Adam(model.parameters(), lr=hparams.lr)
scheduler = StepLR(optimizer, hparams.step_size, 
                   gamma=hparams.decay_gamma)

# Logs
best_loss = np.inf
ssim_log = []
loss_log = []
coil_log = []
running_loss, running_ssim, running_coil = 0, -1., 0.
local_dir = global_dir + '/%s' % hparams.loss_space
if not os.path.isdir(local_dir):
    os.makedirs(local_dir)

# Preload from the same model hyperparameters
if hparams.start_epoch > 0:
    contents = torch.load(local_dir + '/ckpt_epoch%d.pt' % (hparams.start_epoch-1))
    model.load_state_dict(contents['model_state_dict'])
    optimizer.load_state_dict(contents['optimizer_state_dict'])
    # Increment scheduler
    scheduler.last_epoch = hparams.start_epoch-1

# For each epoch
for epoch_idx in range(hparams.start_epoch, hparams.num_epochs):
    # For each batch
    for sample_idx, sample in tqdm(enumerate(train_loader)):
        # Move to CUDA
        for key in sample.keys():
            try:
                sample[key] = sample[key].cuda()
            except:
                pass
            
        # Get outputs
        est_img_kernel, est_map_kernel, est_ksp = \
            model(sample, hparams.num_unrolls)
        
        # Get target image
        if hparams.loss_space == 'mvue':
            est_output = torch.abs(crop(est_img_kernel, 384, 384))
            gt_image   = torch.abs(crop(sample['ref_mvue'], 384, 384))
            data_range = sample['data_range_mvue']
        elif hparams.loss_space == 'rss':
            est_img_coils = ifft(est_ksp)
            est_output    = torch.sqrt(torch.sum(torch.square(
                    torch.abs(est_img_coils)), axis=1))
            est_output    = crop(est_output, 384, 384)
            gt_image      = sample['ref_rss']
            data_range    = sample['data_range']
            
        # SSIM loss in image domain
        loss = ssim(est_output[:, None], gt_image[:, None], data_range)
        # Keep a running loss
        running_loss = 0.99 * running_loss + 0.01 * loss.item() if running_loss > 0. else loss.item()
        loss_log.append(running_loss)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        # For MoDL (?), clip gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), hparams.grad_clip)
        optimizer.step()
        
        # Verbose
        print('Epoch %d, Step %d, Batch loss %.4f. Avg. Loss %.4f' % (
            epoch_idx, sample_idx, loss.item(), running_loss))
        
    # Save models
    last_weights = local_dir +'/ckpt_epoch%d.pt' % epoch_idx
    torch.save({
        'epoch': epoch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_log': loss_log,
        'loss': loss,
        'hparams': hparams}, last_weights)
    
    # Scheduler
    scheduler.step()