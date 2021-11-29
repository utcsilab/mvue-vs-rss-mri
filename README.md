# mvue-vs-rss-mri
The truth matters: A brief discussion on MVUE vs. RSS in MRI reconstruction

# Quantitative Results
(reproducible by running plot_results.py)
Average test SSIM (between 0-1, higher is better) evaluated against both MVUE and RSS at equispaced accelerations [4, 8].

```
+--------------+-------------------------+-------------------------+-------------------------+-------------------------+
|              | MoDL-MVUE               | MoDL-RSS                | SENSE-L1                | ZF                      |
+==============+=========================+=========================+=========================+=========================+
| Test on MVUE | [0.9507042  0.89198264] | [0.77505324 0.71675273] | [0.92924458 0.75762608] | [0.78043904 0.6316088 ] |
+--------------+-------------------------+-------------------------+-------------------------+-------------------------+
| Test on RSS  | [0.7820196  0.72397183] | [0.94513266 0.89577562] | [0.7516844  0.66886452] | [0.79385164 0.66280913] |
+--------------+-------------------------+-------------------------+-------------------------+-------------------------+
```

Average test PSNR (in dB, higher is better) evaluated against both MVUE and RSS at equispaced accelerations [4, 8].

```
+--------------+---------------------------+---------------------------+---------------------------+-----------+
|              | MoDL-MVUE                 | MoDL-RSS                  | SENSE-L1                  | ZF        |
+==============+===========================+===========================+===========================+===========+
| Test on MVUE | [38.339743 31.375493] | [33.184236 29.535552] | [37.628838 26.824533] | [27.013849 22.536034] |
+--------------+---------------------------+---------------------------+---------------------------+-----------+
| Test on RSS  | [33.745713 30.013556] | [37.423528 31.244437] | [33.204420 26.754410] | [33.204420 26.754410] |
+--------------+---------------------------+---------------------------+---------------------------+-----------+
```
