import os, h5py, glob
import torch, itertools, argparse
import numpy as np
import sigpy as sp

from tqdm import tqdm
from datagen import TruncatedMVUE
from dotmap import DotMap

from aux_unrolled_cplx_modl import MoDLDoubleUnroll
# Global seed
global_seed = 2000

# Loss modules
nmse_loss = lambda x, y: np.sum(np.square(np.abs(x - y))) /\
    np.sum(np.square(np.abs(x)))
from skimage.metrics import structural_similarity as ssim_loss
from skimage.metrics import peak_signal_noise_ratio as psnr_loss

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--out', type=str, default='rss')
args = parser.parse_args()

# GPU nanagement
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# Turn off reduced precision stuff
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True

# Dummy dataloader - used to get masks
val_dataset   = TruncatedMVUE([], [], [], [])

# Method and dataset
target_out   = args.out
target_epoch = 14
method       = 'modl_%s_epoch%d' % (target_out, target_epoch)
dataset      = 'brain'

# Target model
state_dict_file = 'models_%s/\
UseACS1_CropRSS1_MapSize15x9_MapHeads0_\
lamTrain1/N6_n0_ACSlines0_Blocks4_lr2.0e-04_image/\
ckpt_epoch%d.pt' % (target_out, target_epoch)
# Architecture parameters !!! Will dictate model
contents    = torch.load(state_dict_file)
hparams     = contents['hparams']
model_state = contents['model_state_dict']
# Get the model
model = MoDLDoubleUnroll(hparams).cuda()
# !!! Load weights
model.load_state_dict(model_state, strict=True)
model.eval()

# Get list of files
# Where is the data and the maps
raw_dir = '/your/test/data/here'
map_dir = '/your/test/maps/here'
# Where is the list of test samples (subset of test data, needs a separate folder)
lookup_dir = '/your/subset/test/files/here'
# Get core files and slices
all_files = sorted(glob.glob(lookup_dir + '/*.h5'))
# Create lists
target_files = []
target_slice = []
gt_mvue, s_maps = [], []
kspace, gt_rss  = [], []
# Populate them
for file in all_files:
    core, num = os.path.basename(file).rsplit('_', 1)
    target_files.append(core + '.h5')
    target_slice.append(int(num[:-3]))
    with h5py.File(os.path.join(map_dir, target_files[-1]), 'r') as contents:
        s_maps.append(np.asarray(contents['s_maps'][target_slice[-1]]))
    # Get raw data
    with h5py.File(os.path.join(raw_dir, target_files[-1]), 'r') as contents:
        kspace.append(np.asarray(contents['kspace'][target_slice[-1]]))
        gt_rss.append(np.asarray(contents['reconstruction_rss'][target_slice[-1]]))
    # Estimate MVUE
    local_mvue = np.sum(sp.ifft(kspace[-1], axes=(-1, -2)) *
                        np.conj(s_maps[-1]), axis=0)
    gt_mvue.append(sp.resize(local_mvue, (384, 384)))
    
# Testing hyperparameters
hparams = DotMap()
hparams.downsampling = [4, 8]
hparams.mask_type    = ['equispaced']
hparams.mask_dir     = ['y']
# Number of samples
num_samples = len(kspace)

# Global result directory
global_dir = './results/%s/%s' % (method, dataset)
if not os.path.exists(global_dir):
    os.makedirs(global_dir)
    
# For each hyperparameter
for mask_type, mask_dir in itertools.product(hparams.mask_type, hparams.mask_dir):
    # Local results
    rss_ssim  = np.zeros((len(hparams.downsampling), num_samples))
    rss_psnr  = np.zeros((len(hparams.downsampling), num_samples))
    rss_nmse  = np.zeros((len(hparams.downsampling), num_samples))
    mvue_ssim = np.zeros((len(hparams.downsampling), num_samples))
    mvue_psnr = np.zeros((len(hparams.downsampling), num_samples))
    # Meta-file name
    global_file = global_dir + '/%s_%s.h5' % (mask_type, mask_dir)
    
    # For each acceleration factor
    for ds_idx, ds in tqdm(enumerate(hparams.downsampling)):
        # Sub-file name
        local_dir  = global_dir + '/recons_%s_%s' % (mask_type, mask_dir)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            
        local_file = local_dir + '/R%.2f.h5' % ds
        # Storage
        saved_rec_out, = []
        saved_gt_mvue, saved_gt_rss    = [], []
        
        # For each sample
        for sample_idx in tqdm(range(num_samples)):
            # Get local variables
            local_ksp = kspace[sample_idx]
            local_mps = s_maps[sample_idx]
            local_gt_mvue = gt_mvue[sample_idx]
            local_gt_rss  = gt_rss[sample_idx]
            
            # How many lines reserved as ACS
            if mask_type == 'random' or mask_type == 'equispaced':
                acs_size = 0.08 if ds < 8 else 0.04
                # Get and apply mask
                if mask_dir == 'y':
                    central_lines = int(np.floor(local_ksp.shape[-1] * acs_size))
                    mask          = val_dataset._get_mask(
                        acs_lines=central_lines, total_lines=int(local_ksp.shape[-1]),
                        R=ds, pattern=mask_type)
                    
                    # Mask k-space and estimate maps
                    under_ksp = local_ksp * mask[None, None, ...]
                    # Need reals
                    real_under_ksp = np.concatenate((
                        np.real(under_ksp)[..., None], np.imag(under_ksp)[..., None]),
                        axis=-1).astype(np.float32)
                    
                    # Fill in mask to the right sizes
                    mask = np.repeat(mask[None, ...], under_ksp.shape[1], axis=0)
                    
                    # Create fake sample
                    sample = {'ksp': real_under_ksp,
                              'mask': mask.astype(np.float32),
                              's_maps_cplx': local_mps}
                elif mask_dir == 'x':
                    central_lines = int(np.floor(local_ksp.shape[-2] * acs_size))
                    mask          = val_dataset._get_mask(
                        acs_lines=central_lines, total_lines=int(local_ksp.shape[-2]),
                        R=ds, pattern=mask_type)
                    
                    # Mask k-space and estimate maps
                    under_ksp = local_ksp * mask[None, ..., None]
                    # Need reals
                    real_under_ksp = np.concatenate((
                        np.real(under_ksp)[..., None], np.imag(under_ksp)[..., None]),
                        axis=-1).astype(np.float32)
                    
                    # Fill in mask to the right sizes
                    mask = np.repeat(mask[:, None], under_ksp.shape[2], axis=1)
                    
                    # Create fake sample
                    sample = {'ksp': real_under_ksp,
                              'mask': mask.astype(np.float32),
                              's_maps_cplx': local_mps}
            
            # Move to CUDA
            for key, value in sample.items():
                sample[key] = torch.tensor(sample[key])[None, ...].cuda()
            
            # Get outputs
            with torch.no_grad():
                est_img_kernel, est_maps_kernel, est_ksp = \
                    model(sample, 6)
                # Move to CPU
                est_ksp = est_ksp.cpu().detach().numpy()[0]
                
            # Get output signal
            if target_out == 'mvue':
                # Directly as 'x'
                est_out = est_img_kernel.cpu().detach().numpy()[0]
                est_out = sp.resize(np.abs(est_out),
                        local_gt_mvue.shape).astype(np.float32)
            elif target_out == 'rss':
                # Get coils
                complex_est_coils = sp.ifft(est_ksp, axes=(-1, -2))
                # Estimated MVUE and RSS
                est_out = sp.rss(complex_est_coils, axes=(0,))
                est_out = sp.resize(est_out, local_gt_rss.shape).astype(np.float32)
            
            # Evaluate output against both signals
            rss_ssim[ds_idx, sample_idx] = \
                ssim_loss(np.abs(local_gt_rss), np.abs(est_out),
                          data_range=np.abs(local_gt_rss).max())
            rss_psnr[ds_idx, sample_idx] = \
                psnr_loss(np.abs(local_gt_rss), np.abs(est_out),
                     data_range=np.abs(local_gt_rss).max())
            rss_nmse[ds_idx, sample_idx] = nmse_loss(np.abs(local_gt_rss),
                         np.abs(est_out))
            mvue_ssim[ds_idx, sample_idx] = \
                ssim_loss(np.abs(local_gt_mvue), np.abs(est_out),
                          data_range=np.abs(local_gt_mvue).max())
            mvue_psnr[ds_idx, sample_idx] = \
                psnr_loss(np.abs(local_gt_mvue), np.abs(est_out),
                     data_range=np.abs(local_gt_mvue).max())
            mvue_nmse[ds_idx, sample_idx] = nmse_loss(np.abs(local_gt_mvue),
                         np.abs(est_out))
            
            # Store recons and GTs
            saved_rec_out.append(est_out)
            saved_gt_mvue.append(local_gt_mvue)
            saved_gt_rss.append(local_gt_rss)
    
        # Save exact recons for a particular acceleration
        with h5py.File(local_file, 'w') as hf:
            hf.create_dataset('saved_rec_out', data=np.asarray(saved_rec_out).astype(np.float32))
            hf.create_dataset('saved_gt_rss', data=np.asarray(saved_gt_rss).astype(np.float32))
            hf.create_dataset('saved_gt_mvue', data=np.asarray(saved_gt_mvue).astype(np.complex64))
            hf.create_dataset('target_files', data=target_files)
            hf.create_dataset('target_slice', data=target_slice)
            
    # Save meta-results to file
    with h5py.File(global_file, 'w') as hf:
        hf.create_dataset('rss_ssim', data=rss_ssim)
        hf.create_dataset('rss_psnr', data=rss_psnr)
        hf.create_dataset('rss_nmse', data=rss_nmse)
        hf.create_dataset('mvue_ssim', data=mvue_ssim)
        hf.create_dataset('mvue_psnr', data=mvue_psnr)
        hf.create_dataset('mvue_nmse', data=mvue_nmse)

        hf.create_dataset('target_files', data=target_files)
        hf.create_dataset('target_slice', data=target_slice)