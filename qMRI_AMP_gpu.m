%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1. define dataset and algorithm paramters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_file_loc = "update_data_folder_location";  % just the folder locations since there are multiple data files saved in the folder, it must end with "/". In this folder the raw k-space data acquared at the i-th echo time and j-th flip angle should be saved in a 4d array named "data" whose size is sx by sy by sz by Nc, this 4d array shoule by saved into a MAT file named "echo_i_fa_j.mat" (note that i and j should be replaced by actual numbers) 
sampling_pattern_loc = "update_sampling_pattern_file_location"; % the sampling pattern along the sy-sz plane, should be saved in a 2d binary array named 'sampling_pattern' whose size is sy by sz, the sampling pattern should match the acquired raw k-space data
sensitivity_map_loc = "update_sensitivity_map_file_location";  % the estimated sensitivity map, should be saved in a 4d array named "maps_3d" whose size is sx by sy by sz by Nc 
flip_angle_map_loc = "update_flip_angle_map_location";  % the 3D flip angle map that has been corrected to compensate for B0 field inhomogeneity, should be saved in a 3d array named "FA_scaled" whose size is sx by sy by sz
output_file_loc = "update_output_folder_location";   % just the folder location where the output will be saved, it must end with "/"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% update the following dataset parameters accordingly %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

gpu_num = 1;    % the gpu device to be selected, note that the gpu index starts from 1
gpu_device = gpuDevice(gpu_num);    % gpu_device needs to be reset when the job is done to avoid memory overflow

load(sampling_pattern_loc)
num_sampling_loc = length(sampling_pattern(sampling_pattern>0));   % the number of sampling locations in the phase encoding y-z plane, it should be the same across different echo times and flip angles

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
max_pe_est_ite = 5; % the number of iterations of the second-order method to estimate the hyperparameters

max_spar_ite = 20; % the maximum number of iterations during the initialization stage where only sparse prior is used to recover the varialbe-flip-angle multi-echo images
max_spar_mod_ite = 5;      %  the maximum number of alternating iterations between enforcing the signal model prior and enforcing the sparse priors to recover the tissue parameters
max_search_ite = 20;    % the maximum number of (inner) iterations to update the sparse prior or to perform the exhaustive search of tissue parameters to fit the signal model

% define the exhaustive search intervals of tissue parameters 
% they can be determined based on the least square solutions of VFA multi-echo images
% make sure to update them based on your dataset
r2star_max = 0.25;  % the maximum r2star value
r2star_min = 0;     % the minimum r2star value
t1_max = 10000;     % the maximum t1 value
t1_min = 0;         % the minimum t1 value
x0_max = 1;         % the maximum proton density value
x0_min = eps;       % the minimun proton density value, set to eps to avoid NAN values during optimization
exhaustive_search_number = 1001;    % the number of equally spaced quantitative values to be searched between the above specified minimum and maximum quantitative values.

%%%%%%%% define the dictionary used in the exhaustive search of r2star values for the nonlinear least square fitting of the signal model
% create dictionary for estimation of r2star
r2_dict_val = linspace(0,r2star_max,exhaustive_search_number);
r2_exp_dict = zeros(Ne, length(r2_dict_val));
for (i=1:length(r2_dict_val))
    r2_exp_dict(:,i) = exp(-echo_time*r2_dict_val(i));
end 
r2_exp_dict_sq_norm = sum(r2_exp_dict.^2,1);


%%%%%%%% define the dictionary used in the exhaustive search of T1 values for the nonlinear least square fitting of the signal model
% create dictionary for estimation of r2star
t1_dict_val = linspace(0,t1_max,exhaustive_search_number);
t1_dict_exp_val = exp(-TR./t1_dict_val);
t1_exp_dict = t1_dict_exp_val;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2. read data and start reconstruction %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% save the sampling vectors at different echo times and flip angles
loc_index = reshape(1:(sy*sz), [sy sz]);    % location indices

sampling_vect = zeros(num_sampling_loc, Ne, Nf); % sampling vectors
noisy_measure = zeros(sx, num_sampling_loc, Nc, Ne, Nf); % fourier measurements

% read the k-space raw data
% the sampling vectors (locations) are derived from the nonzero locations in the data
for (echo_idx = 1:Ne) 
for (fa_idx = 1:Nf)

    % the measurments are saved in a MAT file containing a variable "data", which is a matrix of size "sx by sy by sz by Nc"
    load(strcat(data_file_loc, 'echo_', num2str(echo_idx), '_fa_', num2str(fa_idx), '.mat'))

    % find the nonzero sampling locations
    sampling_vect_tmp = sort(loc_index(sampling_pattern==1));

    sampling_vect(:,echo_idx,fa_idx) = sampling_vect_tmp;

    % read the fourier measurements to noisy_measure
    for (j=1:Nc)
        for (k=1:sx)
            X_fft_noisy_tmp = squeeze(data(k,:,:,j));
            noisy_measure(k,:,j,echo_idx,fa_idx) = X_fft_noisy_tmp(sampling_vect(:,echo_idx,fa_idx));
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% use conjungate gradient descent to find the least square solution  %%
%% and use it to estimate the range of image magnitude (maxmimum and  %%
%% minimum pixel values)                                              %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_echo = noisy_measure;
clear noisy_measure;

% use conjungate gradient descent to find the least square solution with minimum l2 norm
mat_sz = [sx sy sz];

X_init = zeros(sx,sy,sz,Ne,Nf);

    for (f=1:Nf)

    X_init_Nf = zeros(sx, sy, sz, Ne);

    for (j=1:Ne)
        X_init_tmp = gpuArray(zeros(sx, sy, sz));
        d_cg = -sx*sy*sz*At_op_3d( conj(maps_3d), A_op_3d(maps_3d.*X_init_tmp, sampling_vect(:,j,f)) - y_echo(:,:,:,j,f), sampling_vect(:,j,f) );
        r_cg = d_cg;
        for (ite=1:max_cg_ite)   % max iteration is 20
            a_cg_n = sum(conj(r_cg).*r_cg, 'all');
            a_cg_d = sum(sx*sy*sz*(At_op_3d( conj(d_cg).*conj(maps_3d), A_op_3d(maps_3d.*d_cg, sampling_vect(:,j,f)) -y_echo(:,:,:,j,f), sampling_vect(:,j,f) )), 'all');

            a_cg_d = real(a_cg_d);
            a_cg = a_cg_n / a_cg_d;
            X_init_tmp_pre = X_init_tmp;
            X_init_tmp = X_init_tmp + a_cg * d_cg;
            cvg_cg_val = norm(X_init_tmp(:)-X_init_tmp_pre(:), 'fro')/norm(X_init_tmp(:), 'fro');
            fprintf('Echo %d, cvg val %d\n', j, cvg_cg_val)
            if (cvg_cg_val<cg_tol)
                break;
            end
            r_cg_new = r_cg - a_cg*sx*sy*sz*At_op_3d(conj(maps_3d), A_op_3d(maps_3d.*d_cg, sampling_vect(:,j,f)) - y_echo(:,:,:,j,f), sampling_vect(:,j,f) );
            b_cg = sum(conj(r_cg_new).*r_cg_new, 'all')/sum(conj(r_cg).*r_cg, 'all');
            d_cg = r_cg_new + b_cg*d_cg;
            r_cg = r_cg_new;
        end
        X_init_Nf(:,:,:,j) = X_init_tmp;
    end 

    X_init(:,:,:,:,f) = X_init_Nf;

    end

X_init(abs(X_init)==0) = eps;
clear X_init_mat X_init_Nf X_init_tmp X_init_tmp_pre d_cg r_cg r_cg_new;
reset(gpu_device);  % reset gpu device to clear memory


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% we need X_init to estimate the range of signal magnitude %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 1. Reconstruct the variable-flip-angle multi-echo MR %%
%%         images using the sparse prior only. They are used %%
%%         to initialize the model-based AMP-PE framework    %%
%%         that incorporates both the signal model and       %%
%%         sparse priors                                     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set parallel pool

X0_tmp = zeros(sx,sy,sz);
X0_psi_tmp = Psi(X0_tmp);
X0_psi_tmp = extract_3d_wav_coef(X0_psi_tmp);

wav_coef_len = length(X0_psi_tmp);

% initialize the gamp parameters
gamp_par.max_spar_ite = max_spar_ite;
gamp_par.max_pe_est_ite = max_pe_est_ite;

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
hS = @(in1, in2) A_op_3d(in1, in2);
hSt = @(in1, in2, in3) At_op_3d(in1, in2, in3);

E_coef = @(in) extract_3d_wav_coef(in);
C_struct = @(in) construct_3d_wav_struct(in, wav_vessel);

% the measurement operator
M=num_sampling_loc*sx*Ne*Nc*Nf;
N=wav_coef_len*Ne*Nf;
A_2_echo_fa_comp_3d = A_2_echo_fa_comp_3d_LinTrans(M,N,maps_3d,mat_sz,num_sampling_loc,Ne,Nc,Nf,wav_coef_len,sampling_vect,Psit,Psi,C_struct,E_coef,hS,hSt);

% reset gpu device
reset(gpu_device)

% reconstruct VFA multi-echo images using only the sparse prior for initialization
[res, input_par_new, output_par_new] = gamp_mri_x0_r2star_comp_fa_echo_3d_spar(A_2_echo_fa_comp_3d, y_echo, gamp_par, input_par, output_par);

reset(gpu_device)

%save(strcat(output_file_loc, '_s_res'), 'res', '-v7.3')
%save(strcat(output_file_loc, '_s_input_par_new'), 'input_par_new')
%save(strcat(output_file_loc, '_s_output_par_new'), 'output_par_new')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 2. Reconstruct quantitative MR maps through AMP-PE %%
%%         that incorporates both the signal model and     %%
%%         sparse prior                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% update the initialization parameters for tissue parameter estimation
gamp_par.x_hat_meas_psi = res.x_hat_meas_psi;
gamp_par.tau_x_meas_psi = res.tau_x_meas_psi(1);
gamp_par.s_hat_meas_1 = res.s_hat_meas_1;
gamp_par.s_hat_meas_2 = res.s_hat_meas_1;

output_par.tau_w_1 = output_par_new.tau_w_1;
output_par.tau_w_2 = output_par_new.tau_w_1;

% clear res to save memory
clear res;

M=sx*num_sampling_loc*Ne*Nc*Nf;
N=sx*sy*sz*Ne*Nf;
% nw means no wavelet
A_2_echo_nw_fa_comp_3d = A_2_echo_nw_fa_comp_3d_LinTrans(M,N,maps_3d,mat_sz,num_sampling_loc,Ne,Nc,Nf,sampling_vect,hS,hSt);

%save memory
reset(gpu_device)
clear maps_3d;

M=sx*sy*sz*Ne*Nf;
N=wav_coef_len*Ne*Nf;
A_wav_fa_3d = A_wav_fa_3d_LinTrans(M,N,mat_sz,wav_coef_len,Ne,Nf,Psit,Psi,C_struct,E_coef);

M=sx*sy*sz;
N=wav_coef_len;
A_wav_single_3d = A_wav_single_3d_LinTrans(M,N,mat_sz,wav_coef_len,Psit,Psi,C_struct,E_coef);

% it is important to NOT initialize the exp part from results of the spar part, this creauts some hallow shapes in the reconstructed images
% initialize the recovered multi-echo images from the previous step
x_hat_init = A_wav_fa_3d.mult(gamp_par.x_hat_meas_psi);

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
gamp_par.damp_rate = damp_rate;

gamp_par.max_spar_mod_ite = max_spar_mod_ite;
gamp_par.max_search_ite = max_search_ite;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the decoupled recovery based on the  %%
%% VFA multi-echo images computed in the first  %%
%% step to initialize the quantitative maps for %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% gpu is slower for these types of operations

% set parallel pool
parpool(LN_num, 'IdleTimeout', Inf)
% First estimate the T1
warning('off','all')
ET1 = zeros(sx,sy,sz);
parfor (idx_x=1:sx) % parallel computing need to be set at the outer loop other wise it will be too slow
    fprintf('%d\n',idx_x)
    warning('off','all')
    for (idx_y=1:sy)
        for (idx_z=1:sz)
            T_mat = zeros(Nf*Ne,Ne+1);
            s_sin_seq = zeros(Nf*Ne,1);
            for (j=1:Ne)
                s_seq = abs(squeeze(x_hat_init(idx_x,idx_y,idx_z,j,:)));
                fa_tan_seq = squeeze(tan_FA_scaled(idx_x,idx_y,idx_z,:));
                fa_sin_seq = squeeze(sin_FA_scaled(idx_x,idx_y,idx_z,:));
                T_mat(((j-1)*Nf+1):(j*Nf),1) = s_seq./fa_tan_seq;
                T_mat(((j-1)*Nf+1):(j*Nf),1+j) = 1;
                s_sin_seq(((j-1)*Nf+1):(j*Nf)) = s_seq./fa_sin_seq;
            end
            T_mat = inv(T_mat'*T_mat)*T_mat';
            ET1_PDStar_seq = T_mat*s_sin_seq;
            ET1(idx_x,idx_y,idx_z) = ET1_PDStar_seq(1);
        end
    end
end

ET1 = min(ET1, 1-eps);
ET1 = max(ET1, eps);


% Finally Estimate the PD and R2Star
Et = abs(x_hat_init);
Et(Et<eps) = eps;
Et_log = log(Et);

H = zeros(sx,sy,sz);
R2_star = zeros(sx,sy,sz);
parfor (idx_x=1:sx) % parallel computing need to be set at the outer loop other wise it will be too slow
    fprintf('%d\n',idx_x)
    warning('off','all')
    for (idx_y=1:sy)
        for (idx_z=1:sz)
            T_mat = [repmat(1, Ne*Nf, 1) repmat(-echo_time, [Nf 1])];
            Et_tmp = squeeze(Et(idx_x,idx_y,idx_z,:,:));
            Et_tmp = Et_tmp(:);
            T_mat(:,1) = T_mat(:,1).*Et_tmp;
            T_mat(:,2) = T_mat(:,2).*Et_tmp;

            Et_log_tmp_1 = squeeze(Et_log(idx_x,idx_y,idx_z,:,:));
            Et_log_tmp_1 = Et_log_tmp_1(:);
            Et_log_tmp_2_tmp = (squeeze(sin_FA_scaled(idx_x,idx_y,idx_z,:))*(1-ET1(idx_x,idx_y,idx_z)))./(1-squeeze(cos_FA_scaled(idx_x,idx_y,idx_z,:))*ET1(idx_x,idx_y,idx_z));
            Et_log_tmp_2 = zeros(Ne*Nf, 1);
            for (f=1:Nf)
                Et_log_tmp_2(((f-1)*Ne+1):(f*Ne),1) = repmat(Et_log_tmp_2_tmp(f), [Ne 1]);
            end
            Et_log_tmp_2(Et_log_tmp_2<eps) = eps;
            Et_log_tmp_2 = log(Et_log_tmp_2);
            Et_log_tmp = Et_log_tmp_1-Et_log_tmp_2;
            Et_log_tmp = Et_log_tmp.*Et_tmp;
            T_mat=inv(T_mat'*T_mat)*T_mat';
            H_R2_star_seq = T_mat*Et_log_tmp;
            H(idx_x,idx_y,idx_z)=H_R2_star_seq(1);
            R2_star(idx_x,idx_y,idx_z)=H_R2_star_seq(2);
        end
    end
end

poolobj = gcp('nocreate');  % close parallel pool
delete(poolobj);
warning('on','all')

% proton density initialization
H_exp = exp(H);
H_exp = min(H_exp, gamp_par.x0_max);
H_exp = max(H_exp, gamp_par.x0_min);

% r2_star initalization
R2_star = min(R2_star, gamp_par.r2star_max);
R2_star = max(R2_star, gamp_par.r2star_min);

% T1 intialization is -1./log(ET1)*TR
% ET1 is used here for simplicity

gamp_par.ET1 = ET1;
gamp_par.H_exp = H_exp;
gamp_par.R2_star = R2_star;

[res, input_par_new, output_par_new] = gamp_mri_x0_r2star_comp_fa_echo_3d_model_spar(A_2_echo_nw_fa_comp_3d, A_wav_fa_3d, A_wav_single_3d, y_echo, gamp_par, input_par, output_par);

T1_rec = -1./log(res.ET1)*TR;   % the recovered T1 map
R2star_rec = res.r2star;        % the recovered R2* map
Z0_rec = res.x0;                % the recovered proton density map 
Z_rec = res.x_hat_all;          % the recovered VFA multi-echo images

save(strcat(output_file_loc, 'rec_qmri'), 'T1_rec', 'R2star_rec', 'Z0_rec', 'Z_rec', '-v7.3')
