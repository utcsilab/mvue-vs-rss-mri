# mvue-vs-rss-mri
The truth matters: A brief discussion on MVUE vs. RSS in MRI reconstruction

# Qualitative Results
(reproducible by running plot_results.py and using pre-saved results available in 'results')

The figure below shows the perils of evaluating on mismatched images: MoDL is trained on MVUE, and evaluated on RSS. PICS (L1-Wavelet reconstruction using SigPy) is tuned for MVUE, and evaluated on RSS. Both images look good, but their score suffers a large, unfair penalty. Zero-filled RSS is objectively better than both in SSIM (and PSNR, not shown here), but fails qualitative inspection.

<img src="images/figure1.png" width="600"/>

# Quantitative Results
(reproducible by running plot_results.py)

Average test SSIM (between 0-1, higher is better) evaluated against both MVUE and RSS at equispaced accelerations [4, 8].

```
+--------------+-----------------+-----------------+-----------------+-----------------+
|              | MoDL-MVUE       | MoDL-RSS        | SENSE-L1        | Zero-Filled     |
+==============+=================+=================+=================+=================+
| Test on MVUE | [0.9507 0.8919] | [0.7750 0.7167] | [0.9292 0.7576] | [0.7804 0.6316] |
+--------------+-----------------+-----------------+-----------------+-----------------+
| Test on RSS  | [0.7820 0.7239] | [0.9451 0.8957] | [0.7516 0.6688] | [0.7938 0.6628] |
+--------------+-----------------+-----------------+-----------------+-----------------+
```

Average test PSNR (in dB, higher is better) evaluated against both MVUE and RSS at equispaced accelerations [4, 8].

```
+--------------+-----------------+-----------------+-----------------+-----------------+
|              | MoDL-MVUE       | MoDL-RSS        | SENSE-L1        | Zero-Filled     |
+==============+=================+=================+=================+=================+
| Test on MVUE | [38.333 31.375] | [33.184 29.535] | [37.628 26.824] | [27.013 22.536] |
+--------------+-----------------+-----------------+-----------------+-----------------+
| Test on RSS  | [33.745 30.013] | [37.423 31.244] | [33.204 26.754] | [33.204 26.754] |
+--------------+-----------------+-----------------+-----------------+-----------------+
```
