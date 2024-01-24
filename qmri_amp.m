    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Section 1. define dataset and algorithm paramters %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    data_file_loc = "update_data_folder_location";  % just the folder locations since there are multiple data files saved in the folder
    sensitivity_map_loc = "update_sensitivity_map_file_location";  % the file location 
    flip_angle_map_loc = "update_flip_angle_map_location";  % the file location
    ouput_file = "update_output_folder_location";   % just the folder location
    

    % dataset parameters
    sx = 256;       % size along readout x-direction
    sy = 232;       % size along phase encoding y-direction
    sz = 96;        % size along phase encoding z-direction
    Nc = 32;        % the number of channels (coils)
    Ne = 4;         % the number of echoes
    Nf = 3;         % the number of flip angles
    echo_time = [7 15 23 31].'; % echo time in ms
    TR = 36;        % repetition time in ms
    mat_sz = [sx sy sz];

    LN_num = 10;                        % computing thread
    LN = maxNumCompThreads( LN_num );   % set the largest number of computing threads


    num_sampling_loc = 2224;   % the number of sampling locations in the phase encoding y-z plane, it should be the same across different echo times and flip angles

    % wavelet transform paramters
    nlevel = 4;     % wavelet transform level, usually 3-4 is enough
    wave_idx = 6;   % specify db1 to db8, use db6 to balance complexity and performance

    % least square via conjungate gradient paramters for the initialization stage
    cg_tol=1e-4;            % cg convergence threshold
    max_cg_ite = 20;        % the number of cg iterations to compute the least square solution of multi-echo images

    %%%%%%%%%%% gamp parameters
    cvg_thd = 1e-6;     % convergence threshold
    kappa = 1;          % damping rate (learning rate) for parameter estimation, decrease this if the algorithm does not converge
    damp_rate = 1;      % damping rate (learning rate) for signal recovery, decrease this if the algorithm does not converge
    max_pe_est_ite = 5; % the number of inner iterations to estimate the hyperparameters

    max_spar_ite = 50; % the maximum number of iterations during the initialization stage where only sparse prior is used to recover the varialbe-flip-angle multi-echo images
    max_spar_mod_ite = 10;      %  the maximum number of AMP iterations to recover the tissue parameters
    max_search_ite = 20;    % the maximum number of iterations during the exhaustive search of tissue parameters to fit the signal model

    % define the exhaustive search intervals of tissue parameters 
    r2star_max = 0.25;  % the maximum r2star value
    r2star_min = 0;     % the minimum r2star value
    t1_max = 10000;     % the maximum t1 value
    t1_min = 0;         % the minimum t1 value
    x0_max = 1;         % the maximum proton density value
    x0_min = eps;       % the minimun proton density value, set to eps to avoid NAN values during optimization
                    

    %%%%%%%% define the dictionary used in the exhaustive search of r2star values for the nonlinear least square fitting of the signal model
    % create dictionary for estimation of r2star
    r2_dict_val = linspace(0,r2star_max,1001);
    r2_exp_dict = zeros(Ne, length(r2_dict_val));
    for (i=1:length(r2_dict_val))
        r2_exp_dict(:,i) = exp(-echo_time*r2_dict_val(i));
    end 
    r2_exp_dict_sq_norm = sum(r2_exp_dict.^2,1);


    %%%%%%%% define the dictionary used in the exhaustive search of T1 values for the nonlinear least square fitting of the signal model
    % create dictionary for estimation of r2star
    t1_dict_val = linspace(0,t1_max,1001);
    t1_dict_exp_val = exp(-TR./t1_dict_val);
    t1_exp_dict = t1_dict_exp_val;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Section 2. read data and start reconstruction %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % save the sampling vectors at different echo times and flip angles
    loc_index = reshape(1:(sy*sz), [sy sz]);    % location indices

    sampling_vect = zeros(num_sampling_loc, Ne, Nf); % sampling vectors
    noisy_measure = zeros(sx, num_sampling_loc, Ne, Nc, Nf); % fourier measurements


    % the sampling vectors (locations) are derived from the nonzero locations in the data
    for (echo_idx = 1:Ne) 
    for (fa_idx = 1:Nf)

        % the measurments are saved in a MAT file containing a variable "data", which is a matrix of size "sx by sy by sz by Nc"
        load(strcat(data_file_loc, 'echo_', num2str(echo_idx), '_fa_', num2str(fa_idx), '.mat'))

        % find the nonzero sampling locations
        sampling_loc = sum(abs(data),4);
        sampling_loc = squeeze(sum(sampling_loc,1));
        sampling_loc(sampling_loc>0) = 1;
        sampling_vect_tmp = sort(loc_index(sampling_loc==1));

        sampling_vect(:,echo_idx,fa_idx) = sampling_vect_tmp;

        % read the fourier measurements to noisy_measure
        for (j=1:Nc)
            for (k=1:sx)
                X_fft_noisy_tmp = squeeze(data(k,:,:,j));
                noisy_measure(k,:,echo_idx,j,fa_idx) = X_fft_noisy_tmp(sampling_vect(:,echo_idx,fa_idx));
            end
        end
        clear data;
        
    end
    end


    % read the sensitivity 3D maps estiamted via espirit %%
    load(sensitivity_map_loc); % the sensitivity map is saved in to a variable "map_3d", which is a matrix of size "sx by sy by sz by Nc"

    % read the scaled flip angle map
    load(flip_angle_map_loc)
    sin_FA_scaled = sin(FA_scaled);
    tan_FA_scaled = tan(FA_scaled);
    cos_FA_scaled = cos(FA_scaled);

    % use wavelet transform to enforce the sparsity of wavelet coefficients

    X0 = zeros(sx, sy, sz);

    % construct the sensing matrix
    dwtmode('per');
    C1=wavedec3(X0,nlevel,'db1'); 
    ncoef1=length(C1.dec);
    C2=wavedec3(X0,nlevel,'db2'); 
    ncoef2=length(C2.dec);
    C3=wavedec3(X0,nlevel,'db3'); 
    ncoef3=length(C3.dec);
    C4=wavedec3(X0,nlevel,'db4'); 
    ncoef4=length(C4.dec);
    C5=wavedec3(X0,nlevel,'db5'); 
    ncoef5=length(C5.dec);
    C6=wavedec3(X0,nlevel,'db6'); 
    ncoef6=length(C6.dec);
    C7=wavedec3(X0,nlevel,'db7'); 
    ncoef7=length(C7.dec);
    C8=wavedec3(X0,nlevel,'db8'); 
    ncoef8=length(C8.dec);


    switch wave_idx
    case 1
        Psi = @(x) [wavedec3(x,nlevel,'db1')']; 
        Psit = @(x) (waverec3(x));
        wav_vessel = C1;
    case 2
        Psi = @(x) [wavedec3(x,nlevel,'db2')']; 
        Psit = @(x) (waverec3(x));
        wav_vessel = C2;
    case 3
        Psi = @(x) [wavedec3(x,nlevel,'db3')']; 
        Psit = @(x) (waverec3(x));
        wav_vessel = C3;
    case 4
        Psi = @(x) [wavedec3(x,nlevel,'db4')']; 
        Psit = @(x) (waverec3(x));
        wav_vessel = C4;
    case 5
        Psi = @(x) [wavedec3(x,nlevel,'db5')']; 
        Psit = @(x) (waverec3(x));
        wav_vessel = C5;
    case 6
        Psi = @(x) [wavedec3(x,nlevel,'db6')']; 
        Psit = @(x) (waverec3(x));
        wav_vessel = C6;
    case 7
        Psi = @(x) [wavedec3(x,nlevel,'db7')']; 
        Psit = @(x) (waverec3(x));
        wav_vessel = C7;
    otherwise
        Psi = @(x) [wavedec3(x,nlevel,'db8')']; 
        Psit = @(x) (waverec3(x));
        wav_vessel = C8;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% use conjungate gradient descent to find the least square solution with minimum l2 norm %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    y_echo = noisy_measure;
    clear noisy_measure;

    % use conjungate gradient descent to find the least square solution with minimum l2 norm
    mat_sz = [sx sy sz];

    X_init = zeros(sx,sy,sz,Ne,Nf);
    if (1) 

        if (LN_num<Nf)
            parpool(LN_num)
        else
            parpool(Nf)
        end

        parfor (f=1:Nf)

        X_init_Nf = zeros(sx,sy,sz,Ne);

        for (j=1:Ne)
            d_cg = zeros(sx, sy, sz);
            X_init_tmp = zeros(sx, sy, sz);
            for (k=1:Nc)
                d_cg = d_cg - sx*sy*sz*conj(maps_3d(:,:,:,k)).*At_op_3d_cylinder(A_op_3d_cylinder(maps_3d(:,:,:,k).*X_init_tmp, sampling_vect(:,j,f))-y_echo(:,:,j,k,f), sampling_vect(:,j,f), mat_sz);
            end
            r_cg = d_cg;
            for (ite=1:max_cg_ite)   % max iteration is 100
                a_cg_n = sum(conj(r_cg).*r_cg, 'all');
                a_cg_d = 0;
                for (k=1:Nc)
                    a_cg_d = a_cg_d + sum(sx*sy*sz*(conj(d_cg).*conj(maps_3d(:,:,:,k)).*At_op_3d_cylinder(A_op_3d_cylinder(maps_3d(:,:,:,k).*d_cg, sampling_vect(:,j,f)), sampling_vect(:,j,f), mat_sz)), 'all');
                end
                a_cg_d = real(a_cg_d);
                a_cg = a_cg_n / a_cg_d;
                X_init_tmp_pre = X_init_tmp;
                X_init_tmp = X_init_tmp + a_cg * d_cg;
                cvg_cg_val = norm(X_init_tmp(:)-X_init_tmp_pre(:), 'fro')/norm(X_init_tmp(:), 'fro');
                fprintf('Echo %d, cvg val %d\n', j, cvg_cg_val)
                if (cvg_cg_val<cg_tol)
                    break;
                end
                r_cg_new = r_cg;
                for (k=1:Nc)
                    r_cg_new = r_cg_new - a_cg*sx*sy*sz*conj(maps_3d(:,:,:,k)).*At_op_3d_cylinder(A_op_3d_cylinder(maps_3d(:,:,:,k).*d_cg, sampling_vect(:,j,f)), sampling_vect(:,j,f), mat_sz);
                end
                b_cg = sum(conj(r_cg_new).*r_cg_new, 'all')/sum(conj(r_cg).*r_cg, 'all');
                d_cg = r_cg_new + b_cg*d_cg;
                r_cg = r_cg_new;
            end
            X_init_Nf(:,:,:,j) = X_init_tmp;
        end 

        X_init(:,:,:,:,f) = X_init_Nf;

        end
        poolobj = gcp('nocreate');  % close parallel pool
        delete(poolobj);

        %save(strcat(init_file, '_X_init'), 'X_init')

    end

    X_init(abs(X_init)==0) = eps;


    %%% If the initialization is computed with many iterations of conjungate gradient method, it would produce an initialization that spent a lot of effort minimizing the noise error (thus producing the checkerboard pattern)
    %%% we need to only use them to estimate the distribution parameters, use zero initializations for the variables

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% we need X_init to estimate the pixel range %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% only use the H_exp to initialize the distribution parameters

    % set parallel pool

    X0_tmp = zeros(sx,sy,sz);
    X0_psi_tmp = Psi(X0_tmp);
    X0_psi_tmp = extract_3d_wav_coef(X0_psi_tmp);
    
    wav_coef_len = length(X0_psi_tmp);
    
	% initialize the gamp parameters
    gamp_par.max_spar_ite = max_spar_ite;
    gamp_par.max_pe_est_ite = max_pe_est_ite;

    gamp_par.max_spar_mod_ite = max_spar_mod_ite;
    gamp_par.max_search_ite = max_search_ite;

	gamp_par.cvg_thd = cvg_thd;
	gamp_par.kappa = kappa;
    gamp_par.damp_rate = damp_rate;
    gamp_par.echo_time = echo_time;
    gamp_par.sx = sx;
    gamp_par.sy = sy;
    gamp_par.sz = sz;
    gamp_par.Ne = Ne;
    gamp_par.Nc = Nc;
    gamp_par.Nf = Nf;

    gamp_par.x_mag_max = x0_max;
    gamp_par.x_mag_min = x0_min;
    gamp_par.x0_max = x0_max;
    gamp_par.x0_min = x0_min;
    gamp_par.r2star_max = r2star_max;
    gamp_par.r2star_min = r2star_min;
    gamp_par.t1_max = t1_max;
    gamp_par.t1_min = t1_min;

	
	% initialize the variables
    x_hat_meas = X_init;

    % initialize gamp variables
    x_hat_meas_psi = zeros([wav_coef_len Ne Nf]);
    for (f_idx=1:Nf)
    for (i=1:Ne)
        x_hat_meas_psi_struct_tmp = Psi(x_hat_meas(:,:,:,i,f_idx));
        x_hat_meas_psi(:,i,f_idx) = extract_3d_wav_coef(x_hat_meas_psi_struct_tmp);
    end
    end
    tau_x_meas_psi = var(x_hat_meas_psi(:));

    gamp_par.tau_x_meas_psi = tau_x_meas_psi;
    gamp_par.x_hat_meas_psi = x_hat_meas_psi;
    gamp_par.s_hat_meas_1 = zeros(size(y_echo));

	% initialize the distribution parameters
    lambda_x_hat_psi = zeros(Ne,Nf);
    for (f_idx=1:Nf)
    for (i = 1:Ne)
        x_hat_psi_tmp = x_hat_meas_psi(:,i,f_idx);
        lambda_x_hat_psi(i,f_idx) = 1/sqrt(var(abs(x_hat_psi_tmp))/2);
    end
    end
    input_par.lambda_x_hat_psi = lambda_x_hat_psi;

	% output variance parameter
	output_par.tau_w_1 = 1e-12;
	
	% construct measurement operator
	hS = @(in1, in2) A_op_3d_cylinder(in1, in2);
	hSt = @(in1, in2) At_op_3d_cylinder(in1, in2, mat_sz);

    E_coef = @(in) extract_3d_wav_coef(in);
    C_struct = @(in) construct_3d_wav_struct(in, wav_vessel);

    % the measurement operator
    M=num_sampling_loc*sx*Ne*Nc*Nf;
    N=wav_coef_len*Ne*Nf;
    A_2_echo_fa_comp_3d = A_2_echo_fa_comp_3d_cylinder_LinTrans(M,N,maps_3d,mat_sz,num_sampling_loc,Ne,Nc,Nf,wav_coef_len,sampling_vect,Psit,Psi,C_struct,E_coef,hS,hSt);

    % reconstruct VFA multi-echo images using only the sparse prior for initialization
    [res, input_par_new, output_par_new] = gamp_mri_x0_r2star_comp_fa_echo_l1_3d_spar(A_2_echo_fa_comp_3d, y_echo, gamp_par, input_par, output_par);

    %save(strcat(output_file, '_s_res'), 'res', '-v7.3')
    %save(strcat(output_file, '_s_input_par_new'), 'input_par_new')
    %save(strcat(output_file, '_s_output_par_new'), 'output_par_new')


    % update the initialization parameters for tissue parameter estimation
    gamp_par.x_hat_meas_psi = res.x_hat_meas_psi;
    gamp_par.tau_x_meas_psi = res.tau_x_meas_psi(1);
    gamp_par.s_hat_meas_1 = res.s_hat_meas_1;
    gamp_par.s_hat_meas_2 = res.s_hat_meas_1;

    output_par.tau_w_1 = output_par_new.tau_w_1;
    output_par.tau_w_2 = output_par_new.tau_w_1;

    clear res;

    M=sx*sy*sz*Ne*Nf;
    N=wav_coef_len*Ne*Nf;
    A_wav_fa_3d = A_wav_fa_3d_LinTrans(M,N,mat_sz,wav_coef_len,Ne,Nf,Psit,Psi,C_struct,E_coef);

    % it is important to NOT initialize the exp part from results of the spar part, this creauts some hallow shapes in the reconstructed images
    % initialize the exp part from the least square solution
    gamp_par.x_hat_init = x_hat_meas;
    gamp_par.LN_num = LN_num;

    gamp_par.sin_FA_scaled = sin_FA_scaled;
    gamp_par.tan_FA_scaled = tan_FA_scaled;
    gamp_par.cos_FA_scaled = cos_FA_scaled;

    % initialize dictionary needed for estimating r2star values
    gamp_par.r2_dict_val = r2_dict_val;
    gamp_par.r2_exp_dict = r2_exp_dict;
    gamp_par.r2_exp_dict_sq_norm = r2_exp_dict_sq_norm;

    gamp_par.t1_exp_dict = t1_exp_dict;

    [res, input_par_new, output_par_new] = gamp_mri_x0_r2star_comp_fa_echo_l1_alternating_model_spar_v5(A_2_echo_fa_comp_3d, A_wav_fa_3d, y_echo, gamp_par, input_par, output_par);
    
    T1_rec = -1./log(res.ET1)*TR;   % the recovered T1 map
    R2star_rec = res.r2star;        % the recovered R2* map
    Z0_rec = res.x0;                % the recovered proton density map 
    Z_rec = res.x_hat_all;          % the recovered VFA multi-echo images

    save(strcat(output_file, 'rec_qmri'), 'T1_rec', 'R2star_rec', 'Z0_rec', 'Z_rec' '-v7.3')
