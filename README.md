# Accelerated Model-based T1, T2* and Proton Density Mapping 
An approximate message passing (AMP) framework is proposed to recover T1, T2* and proton density maps from undersampled measurements. The proposed AMP-PE approach combines information from both the sparse prior and signal model prior, it treats the parameters as unknown varialbes and automatically estimates them.

* If you use this code and find it helpful, please cite the above paper. Thanks :smile:
```
@ARTICLE{QSM_AMP_PE:2023,
    author    = {Shuai Huang and James J. Lah and Jason W. Allen and Deqiang Qiu},
    title     = {Model-based T1, T2* and Proton Density Mapping Using a Bayesian Approach with Parameter Estimation and Complementary Undersampling Patterns},
    journal   = {arXiv preprint},
    volume    = {arXiv:2307.02015},
    year      = {2023},
    url       = {https://arxiv.org/abs/2307.02015},
}
```


## Summary
```
    ./src	    -- This folder contains MATLAB files to perform 3D reconstruction using the AMP-PE approach.
    ./data      -- The data folder
    ./result	-- The results folder
```

## Dataset Preparation
The datasets need to be placed in the data folder. We first define the following notations:
```
	sx	-- the number of samples along the x direction (readout direction)
	sy	-- the number of samples along the y direction
	sz	-- the number of samples along the z direction
	Ne	-- the number of echoes
	Nf  -- the number of flip angles
	Nc	-- the nunber of channels (sensitivity coils)
```

* 3D datasets
```
	1) VFA multi-echo datasets are ordered according to the echo and flip-angle indices. Suppose there are 4 echoes and 3 flip angles, there will be "4*3=12" files named: "echo_1_fa_1.mat", "echo_1_fa_2.mat", ... Take the file "echo_1_fa_2.mat" for example, it contains the data flip the 1st echo and 2nd flip angle. It is a MAT-file containing a 4-dimensional array named "data". The size of "data" is "sx by sy by sz by Nc". The undersampling takes place in the phase-encoding sy-sz plane, the readout sx direction is always fully sampled. Note that if a location is not sampled in the sy-sz plane, the corresponding entries in the 4D array "data" will be zero. Otherwise, the corresponding entries in "data" will be nonzero and the quantitative maps are reconstructed from the nonzero measurements. Make sure the data is saved this way since the sampling locations are determined by the nonzero entries in the sy-sz plane.
	2) Sensitivity map dataset is named as "sensitivity_map_3d.mat". It is a MAT-file containing a 4-dimensional array named "maps_3d". The size of "maps_3d" is "sx by sy by sz by Nc".
	3) The calibrated flip angles is saved in a MAT-file "FA_scaled.mat". It contains a 4-D array named "FA_scaled", the size of "FA_scaled" is "sx by sy by sz by Nf". Make sure that the flip angle is calibrated properly using a method like the double flip-angle method.
```



## Usage
You can follow the following steps to run the program. 

`Detailed comments` are in the demo file "qmri_amp_gpu.m".

* Step 0) Prepare the dataset in the correct format as detailed above. It should be easy to organize your own dataset in a similar format.

* Step 1) Update the dataset locations and parameters in Section 1 of the demo file "qmri_amp.m".

* Step 2) Open `MATLAB` and type the following commands into the console:
```
    >> addpath('./src')
    >> qmri_amp_gpu	% this is the demo file, look inside for detail comments
```
* Step 3) The recovered quantitative maps and VFA multi-echo images are saved in the MAT-file "rec_qmri.mat"
