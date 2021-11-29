#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:13:44 2021

@author: marius
"""

import h5py, glob, os
import numpy as np
from matplotlib import pyplot as plt

# Global results
global_results = {'rss': dict(),
                  'mvue': dict(),
                  'l1': dict(),
                  'zf': dict()}

# Load ZF results
target_file = './results/zero_filled/brain/equispaced_y.h5'
with h5py.File(target_file, 'r') as contents:
    # Extract stuff
    for key in contents.keys():
        global_results['zf'][key] = np.asarray(contents[key])

# Load L1 results
target_file = './results/l1_wavelet/brain/equispaced_y.h5'
with h5py.File(target_file, 'r') as contents:
    # Extract stuff
    for key in contents.keys():
        global_results['l1'][key] = np.asarray(contents[key])

# Load RSS results
target_file = './results/modl_rss/brain/equispaced_y.h5'
with h5py.File(target_file, 'r') as contents:
    # Extract stuff
    for key in contents.keys():
        global_results['rss'][key] = np.asarray(contents[key])

# Load MVUE results
target_file = './results/modl_mvue/brain/equispaced_y.h5'
with h5py.File(target_file, 'r') as contents:
    # Extract stuff
    for key in contents.keys():
        global_results['mvue'][key] = np.asarray(contents[key])

# Print SSIM table
from tabulate import tabulate
print(tabulate([['Test on MVUE', 
                 np.mean(global_results['mvue']['mvue_ssim'], axis=-1),
                 np.mean(global_results['rss']['mvue_ssim'], axis=-1),
                 np.mean(global_results['l1']['mvue_ssim'], axis=-1),
                 np.mean(global_results['zf']['mvue_ssim'], axis=-1)],
                ['Test on RSS',
                 np.mean(global_results['mvue']['rss_ssim'], axis=-1),
                 np.mean(global_results['rss']['rss_ssim'], axis=-1),
                 np.mean(global_results['l1']['rss_ssim'], axis=-1),
                 np.mean(global_results['zf']['rss_ssim'], axis=-1)]], 
               ['', 'MoDL-MVUE', 'MoDL-RSS', 'SENSE-L1', 'ZF'],
               tablefmt='grid',
               floatfmt='3f'))
# Print PSNR table
print(tabulate([['Test on MVUE', 
                 np.mean(global_results['mvue']['mvue_psnr'], axis=-1),
                 np.mean(global_results['rss']['mvue_psnr'], axis=-1),
                 np.mean(global_results['l1']['mvue_psnr'], axis=-1),
                 np.mean(global_results['zf']['mvue_psnr'], axis=-1)],
                ['Test on RSS',
                 np.mean(global_results['mvue']['rss_psnr'], axis=-1),
                 np.mean(global_results['rss']['rss_psnr'], axis=-1),
                 np.mean(global_results['l1']['rss_psnr'], axis=-1),
                 np.mean(global_results['l1']['rss_psnr'], axis=-1)]], 
               ['', 'MoDL-MVUE', 'MoDL-RSS', 'SENSE-L1', 'ZF'],
               tablefmt='grid',
               floatfmt='3f'))

# Plot an example (default is the one in the abstract)        
target_sample = 3
target_accel  = 4
target_mvue_file = \
    './results/modl_mvue/brain/recons_equispaced_y/R%.2f.h5' % target_accel
target_rss_file  = \
    './results/modl_rss/brain/recons_equispaced_y/R%.2f.h5' % target_accel
target_zf_file = \
    './results/zero_filled/brain/recons_equispaced_y/R%.2f.h5' % target_accel
target_cs_file = \
    './results/l1_wavelet/brain/recons_equispaced_y/R%.2f.h5' % target_accel
# Load
with h5py.File(target_mvue_file, 'r') as contents:
    gt_mvue  = np.asarray(contents['saved_gt_mvue'][target_sample])
    rec_mvue = np.asarray(contents['saved_rec_out'][target_sample])
# More load
with h5py.File(target_rss_file, 'r') as contents:
    gt_rss  = np.asarray(contents['saved_gt_rss'][target_sample])
    rec_rss = np.asarray(contents['saved_rec_out'][target_sample])
# Even more load
with h5py.File(target_zf_file, 'r') as contents:
    zf_mvue = np.asarray(contents['saved_mvue_out'][target_sample])
    zf_rss  = np.asarray(contents['saved_rss_out'][target_sample])
# Load CS as well
with h5py.File(target_cs_file, 'r') as contents:
    l1_mvue    = np.asarray(contents['saved_rec_out'][target_sample])
    
# Plot
plt.rcParams['font.size'] = 23
plt.figure(figsize=(9.8, 10))
vmin, vmax = 0, 5e-4
# Annotations
ann_xoff = 350
ann_font = 23
# More annotations
ann_yoff = 600

# Plot ground truths
plt.subplot(2, 2, 1)
plt.imshow(np.flipud(np.abs(gt_rss)), cmap='gray', vmin=vmin,
           vmax=vmax);
plt.axis('off')
plt.title('Ground Truth RSS')

# Plot MoDL recon from opposite model
plt.subplot(2, 2, 2)
plt.imshow(np.flipud(np.abs(rec_rss)), cmap='gray', vmin=vmin,
           vmax=vmax); plt.axis('off')
plt.annotate('SSIM %.3f' % (
    # global_results['rss']['mvue_ssim'][0, target_sample],
    global_results['mvue']['rss_ssim'][0, target_sample]),
        xy=(0, 0), xycoords='axes fraction',
        xytext=(ann_xoff, 0), textcoords='offset pixels',
        horizontalalignment='center',
        verticalalignment='bottom', color='white', fontsize=ann_font)
plt.title('MoDL-MVUE')

# Plot L1-Wavelet
plt.subplot(2, 2, 3)
plt.imshow(np.flipud(np.abs(l1_mvue)), cmap='gray', vmin=vmin,
           vmax=vmax); plt.axis('off')
plt.annotate('SSIM %.3f' % (
    # global_results['rss']['mvue_ssim'][0, target_sample],
    global_results['l1']['rss_ssim'][0, target_sample]),
        xy=(0, 0), xycoords='axes fraction',
        xytext=(ann_xoff, 0), textcoords='offset pixels',
        horizontalalignment='center',
        verticalalignment='bottom', color='white', fontsize=ann_font)
plt.title('PICS')

# Plot zero-filled stuff
plt.subplot(2, 2, 4)
plt.imshow(np.flipud(np.abs(zf_rss)), cmap='gray', vmin=vmin,
           vmax=vmax);
plt.axis('off')
plt.title('Zero-Filled')
plt.annotate('SSIM %.3f' % (
    global_results['zf']['rss_ssim'][0, target_sample]),
        xy=(0, 0), xycoords='axes fraction',
        xytext=(ann_xoff, 0), textcoords='offset pixels',
        horizontalalignment='center',
        verticalalignment='bottom', color='white', fontsize=ann_font)

plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig('figure1.png', dpi=300, bbox_inches='tight',
    pad_inches=0.05)
plt.close()