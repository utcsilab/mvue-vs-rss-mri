#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:19:12 2020

@author: marius
"""

import h5py, os
import numpy as np
import sigpy as sp
import torch

from tqdm import tqdm

from torch.utils.data import Dataset

def crop(variable, tw, th):
    w, h = variable.shape[-2:]
    x1 = int(np.ceil((w - tw) / 2.))
    y1 = int(np.ceil((h - th) / 2.))
    return variable[..., x1:x1+tw, y1:y1+th]

def crop_cplx(variable, tw, th):
    w, h = variable.shape[-3:-1]
    x1 = int(np.ceil((w - tw) / 2.))
    y1 = int(np.ceil((h - th) / 2.))
    return variable[..., x1:x1+tw, y1:y1+th, :]

class MCFullFastMRI(Dataset):
    def __init__(self, sample_list, num_slices,
                 downsample, saved_masks=None,
                 use_acs=True, scramble=False, 
                 acs_lines=4, mps_kernel_shape=None, maps=None,
                 direction='y', mask_type='equispaced'):
        self.saved_masks  = saved_masks
        self.sample_list  = sample_list
        self.num_slices   = num_slices
        self.downsample   = downsample
        # Multiple R values
        self.multi_R      = len(self.downsample) > 1
        self.mps_kernel_shape = mps_kernel_shape
        self.use_acs      = use_acs
        self.acs_lines    = acs_lines
        self.scramble     = scramble
        self.maps         = maps
        self.direction    = direction # Which direction are lines in
        self.mask_type    = mask_type
            
        # Metadata name
        self.meta_name = os.path.basename(os.path.dirname(sample_list[0]))
        self.meta_file = 'meta_%s_line_data.h5' % self.meta_name
        
        # If we're using everything, we need to count all slices
        if self.num_slices == 'all':
            # Access meta-data of each scan to get number of slices 
            self.all_slices = np.zeros((len(self.sample_list,)), dtype=int)
            for idx, file in tqdm(enumerate(self.sample_list)):
                with h5py.File(file, 'r') as data:
                    self.all_slices[idx] = int(data['kspace'].shape[0])
                    
            # Create cumulative index for mapping
            self.slice_mapper = np.cumsum(self.all_slices) - 1 # Counts from '0'
        
        if self.scramble:
            # One time permutation
            self.permute = np.random.permutation(self.__len__())
            self.inv_permute = np.zeros(self.permute.shape)
            self.inv_permute[self.permute] = np.arange(len(self.permute))
        
    def __len__(self):
        return int(np.sum(self.all_slices))
    
    def __getitem__(self, idx):
        # Always permute
        if self.scramble:
            idx = self.permute[idx]
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.num_slices == 'all':
            # First scan for which index is in the valid cumulative range
            sample_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
            # Offset from cumulative range
            slice_idx = int(idx) if sample_idx == 0 else \
                int(idx - self.slice_mapper[sample_idx] + 
                    self.all_slices[sample_idx] - 1)
            map_idx = slice_idx # All maps available
            target_list = self.sample_list
        else:
            # Separate slice and sample
            sample_idx = idx // self.num_slices
            slice_idx  = self.center_slice + \
                np.mod(idx, self.num_slices) - self.num_slices // 2 
            map_idx = np.mod(idx, self.num_slices) # Maps always count from zero
            target_list = self.sample_list
        
        # Load MRI image
        with h5py.File(target_list[sample_idx], 'r') as contents:
            # Get k-space for specific slice
            k_image = np.asarray(contents['kspace'][slice_idx])
            ref_rss = np.asarray(contents['reconstruction_rss'][slice_idx])
            # Store core file
            core_file  = os.path.basename(target_list[sample_idx])
            core_slice = slice_idx
        
        # If desired, load external sensitivity maps
        if not self.maps is None:
            with h5py.File(self.maps[sample_idx], 'r') as contents:
                # Get sensitivity maps for specific slice
                s_maps = np.asarray(contents['s_maps'][map_idx])
                s_maps_full = np.copy(s_maps)
        else:
            s_maps, s_maps_full = np.asarray([0.]), np.asarray([0.])
            
        # Store all GT data
        gt_ksp = np.copy(k_image)

        # How many central and random slices !!! Hardcoded for four
        sampling_axis = -1 if self.direction == 'y' else -2
        num_central_slices = np.round(0.08 * k_image.shape[sampling_axis])
        # Get locations of center and non-center
        if self.use_acs:
            center_slice_idx = np.arange(
                (k_image.shape[sampling_axis] - num_central_slices) // 2,
                (k_image.shape[sampling_axis] + num_central_slices) // 2)
        else:
            # A fixed number of guaranteed lines
            num_central_slices = self.acs_lines
            center_slice_idx = np.arange(
                (k_image.shape[sampling_axis] - num_central_slices) // 2,
                (k_image.shape[sampling_axis] + num_central_slices) // 2)
    
        # Get a local acceleration
        if self.multi_R is False:
            local_R = self.downsample[0]
        else:
            local_R = np.random.choice(self.downsample)
    
        # If downsampling is done, do it in k-space
        if local_R > 1.01:
            random_slice_candidates = np.setdiff1d(np.arange(
                k_image.shape[sampling_axis]), center_slice_idx)
            # If masks are not fed, generate them on the spot
            if self.saved_masks is None:
                if self.mask_type == 'equispaced':
                    # !!! Only supports integer for now
                    random_slice_idx = random_slice_candidates[::int(local_R)]
                elif self.mask_type == 'random':
                    # Pick random lines outside the center location
                    random_slice_idx = np.random.choice(
                        random_slice_candidates,
                        size=(int(k_image.shape[sampling_axis] // local_R) - 
                              len(center_slice_idx)), 
                        replace=False)
                # Create sampling mask and downsampled k-space data
                k_sampling_mask = np.isin(np.arange(k_image.shape[sampling_axis]),
                                          np.hstack((center_slice_idx,
                                                     random_slice_idx)))
            else:
                # Pick the corresponding mask
                k_sampling_mask = self.saved_masks[idx]
                
            # Apply by deletion
            if self.direction == 'y':
                k_image[..., np.logical_not(k_sampling_mask)] = 0.
                k_sampling_mask = k_sampling_mask[None, ...]
            elif self.direction == 'x':
                k_image[..., np.logical_not(k_sampling_mask), :] = 0.
                k_sampling_mask = k_sampling_mask[..., None]
        else:
            # All ones
            k_sampling_mask = np.ones((1, k_image.shape[-1]))
        
        # Get ACS region
        if self.use_acs:
            if self.direction == 'y':
                acs = k_image[..., center_slice_idx.astype(np.int)]
            elif self.direction == 'x':
                acs = k_image[..., center_slice_idx.astype(np.int), :]
        else:
            # Scale w.r.t. ground truth w.l.o.g.
            acs = gt_ksp
        
        # Normalize k-space based on ACS
        max_acs            = np.max(np.abs(acs))
        k_normalized_image = k_image / max_acs
        # Calculate scaled GT RSS
        ref_rss      = ref_rss / max_acs
        gt_ksp       = gt_ksp / max_acs
        data_range   = np.max(ref_rss)
        # And scaled GT MVUE
        ref_mvue     = np.sum(np.conj(s_maps) * sp.ifft(gt_ksp, axes=(-1, -2)),
                              axis=0)
        data_range_mvue = np.max(np.abs(ref_mvue))
        
        # Initial sensitivity maps
        x_coils    = sp.ifft(k_image, axes=(-2, -1))
        x_rss      = np.linalg.norm(x_coils, axis=0, keepdims=True)
        init_maps  = sp.resize(sp.fft(x_coils / x_rss, axes=(-2, -1)),
                               oshape=tuple([k_image.shape[0]]+self.mps_kernel_shape))
        
        # Concatenate on extra dimension
        k_normalized_image = np.stack((
            np.real(k_normalized_image),
            np.imag(k_normalized_image)), axis=-1)
        
        sample = {'ksp': k_normalized_image.astype(np.float32),
                  'gt_ksp': gt_ksp.astype(np.complex64),
                  's_maps_cplx': s_maps.astype(np.complex64),
                  'init_maps': init_maps.astype(np.complex64),
                  'mask': k_sampling_mask.astype(np.float32),
                  'ref_rss': ref_rss.astype(np.float32),
                  'ref_mvue': ref_mvue.astype(np.complex64),
                  'data_range': data_range,
                  'data_range_mvue': data_range_mvue,
                  'core_file': core_file,
                  'core_slice': core_slice,
                  'downsample': local_R}
        
        return sample