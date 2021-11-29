#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:32:32 2021

@author: yanni
"""

from tqdm import tqdm
from torch.utils.data import Dataset
import torch, os, h5py
import numpy as np
import sigpy as sp

class MVU_Estimator(Dataset):
    def __init__(self, file_list,
                 R=1, pattern='random'):
        # Attributes
        self.file_list    = file_list
        self.R            = R
        self.pattern      = pattern
        
        # Access meta-data of each scan to get number of slices 
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in tqdm(enumerate(self.file_list)):
            with h5py.File(file, 'r') as data:
                self.num_slices[idx] = int(data['kspace'].shape[0])
        
        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'
        
    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans
    
    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random', direction='y'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)
        
        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)
        
        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
                                            
        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.
        
        return mask
    
    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))
        
        return x[..., x1:x1+wout, y1:y1+hout]
    
    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        
        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.file_list[scan_idx])
        with h5py.File(raw_file, 'r') as data:
            # Get raw k-space and rss
            kspace = np.asarray(data['kspace'][slice_idx])
            gt_rss = np.asarray(data['reconstruction_rss'][slice_idx])
        
        # Crop extra lines and reduce FoV by half in readout
        kspace = sp.resize(kspace, (
            kspace.shape[0], kspace.shape[1], 384))
    
        # Reduce FoV by half in the readout direction
        kspace = sp.ifft(kspace, axes=(-2,))
        kspace = sp.resize(kspace, (kspace.shape[0], 384,
                                    kspace.shape[2]))
        kspace = sp.fft(kspace, axes=(-2,)) # Back to k-space 
        
        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.
        
        # Scale data
        kspace = kspace / scale_factor
        
        # Compute ACS size based on R factor and sample size
        total_lines = kspace.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)
        
        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)
        
        # Mask k-space
        kspace *= mask
        
        # Convert to reals
        kspace = np.stack((
            np.real(kspace),
            np.imag(kspace)), axis=-1)
        
        # Output
        sample = {
                  'ksp': kspace,
                  'mask': mask,
                  'gt_rss': gt_rss,
                  'data_range': np.max(np.abs(gt_rss)),
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx}
        
        return sample
    
class TruncatedMVUE(Dataset):
    def __init__(self, mvue_list,
                 raw_list,
                 R=1, pattern='random'):
        # Attributes
        self.mvue_list    = mvue_list
        self.raw_list     = raw_list
        self.R            = R
        self.pattern      = pattern
        
        # Access meta-data of each scan to get number of slices 
        self.num_slices = np.zeros((len(self.mvue_list,)), dtype=int)
        for idx, file in tqdm(enumerate(self.mvue_list)):
            with h5py.File(file, 'r') as data:
                self.num_slices[idx] = int(data['mvue'].shape[0])
        
        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'
        
    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans
    
    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)
        
        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)
        
        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            # np.random.seed(1)
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
                                            
        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.
        
        return mask
    
    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))
        
        return x[..., x1:x1+wout, y1:y1+hout]
    
    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        
        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.raw_list[scan_idx])
        with h5py.File(raw_file, 'r') as data:
            # Get raw k-space
            kspace = np.asarray(data['kspace'][slice_idx])
        
        # Crop extra lines and reduce FoV by half in readout
        kspace = sp.resize(kspace, (
            kspace.shape[0], kspace.shape[1], 384))
    
        # Reduce FoV by half in the readout direction
        kspace = sp.ifft(kspace, axes=(-2,))
        kspace = sp.resize(kspace, (kspace.shape[0], 384,
                                    kspace.shape[2]))
        kspace = sp.fft(kspace, axes=(-2,)) # Back to k-space 
        
        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.
        
        # Scale data
        kspace = kspace / scale_factor
        
        # Get ground truth MVUE with ESPiRiT maps
        with h5py.File(self.mvue_list[scan_idx], 'r') as data:
            gt_mvue = np.asarray(data['mvue'][slice_idx])
        
        # Compute ACS size based on R factor and sample size
        total_lines = kspace.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)
        
        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)
        
        # Mask k-space
        gt_ksp  = np.copy(kspace)
        kspace *= mask
        
        # Convert to reals
        kspace = np.stack((
            np.real(kspace),
            np.imag(kspace)), axis=-1)
        
        # Output
        sample = {
                  'ksp': kspace,
                  'mask': mask,
                  'gt_mvue': gt_mvue,
                  'gt_ksp': gt_ksp,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx}
        
        return sample