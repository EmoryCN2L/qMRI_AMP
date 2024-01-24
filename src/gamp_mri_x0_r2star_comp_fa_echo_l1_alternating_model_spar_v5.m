function [res, input_par, output_par] = gamp_mri_full_x0_r2star_l1_3d_nw_com_com_spar_ahead(A_2_echo_fa_comp_3d, A_wav_fa_3d, y, gamp_par, input_par, output_par)

    % use the same model as the sparsity promoting recovery
    % no wavelet in A_2_echo_fa_comp_3d
    % did not embed parameter estimation, move estimation of x_hat_all to the last step
    % now embed parameter estimation

    % seems that i should not subtract info from measurements when passing info from exp to meas, because it will diverge, so in this case i am not subtracting, use new way to compute x_hat_exp and tau_x_exp

    % still diverges, move x_hat_all ahead
    % use damping on both blocks

    % move x_hat_all behind, do not apply regularization on x0_psi, no damping

    % use a bi block message passing one end is x_hat_all 

    % only pass magnitude info to exp block, the variance is automatically estimated, hope this will stop the rescontruction from becoming low resolution

    % enforce positive constraint on the r2star
	% set GAMP parameters
	max_spar_mod_ite = gamp_par.max_spar_mod_ite;
    max_search_ite = gamp_par.max_search_ite;
    max_spar_ite = gamp_par.max_spar_ite;
    max_pe_est_ite = gamp_par.max_pe_est_ite;
	cvg_thd = gamp_par.cvg_thd;
	kappa = gamp_par.kappa;

    tau_x_meas_psi = gamp_par.tau_x_meas_psi;
    x_hat_meas_psi = gamp_par.x_hat_meas_psi;
    s_hat_meas_1 = gamp_par.s_hat_meas_1;
    s_hat_meas_2 = gamp_par.s_hat_meas_2;

    x_hat_init = gamp_par.x_hat_init;

    % set input distribution parameters
    lambda_x_hat_psi = input_par.lambda_x_hat_psi;

    tau_w_1 = output_par.tau_w_1;
    tau_w_2 = output_par.tau_w_2;

    echo_time = gamp_par.echo_time;
    sx = gamp_par.sx;
    sy = gamp_par.sy;
    sz = gamp_par.sz;
    Ne = gamp_par.Ne;
    Nc = gamp_par.Nc;
    Nf = gamp_par.Nf;

    x_mag_max = gamp_par.x_mag_max;
    x_mag_min = gamp_par.x_mag_min;
    x0_max = gamp_par.x0_max;
    x0_min = gamp_par.x0_min;
    r2star_max = gamp_par.r2star_max;
    r2star_min = gamp_par.r2star_min;

    sin_FA_scaled = gamp_par.sin_FA_scaled;
    cos_FA_scaled = gamp_par.cos_FA_scaled;
    tan_FA_scaled = gamp_par.tan_FA_scaled;

    r2_exp_dict = gamp_par.r2_exp_dict;
    r2_exp_dict_sq_norm = gamp_par.r2_exp_dict_sq_norm;
    r2_dict_val = gamp_par.r2_dict_val;

    t1_exp_dict = gamp_par.t1_exp_dict;

    damp_rate = gamp_par.damp_rate;

    LN_num = gamp_par.LN_num;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% initialize tau_w_exp with tau_x_mag_meas from the warm up step %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% first finish the sparse reconstruction %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    tau_x_mag_meas_1 = A_wav_fa_3d.multSq(tau_x_meas_psi);
    x_mag_meas_1_complex = A_wav_fa_3d.mult(x_hat_meas_psi);


    %%%%%%%%%%%%%%%%
    %% use the recovered signal from previous step as the initialization
    %%%%%%%%%%%%%%%%
    x_hat_init = x_mag_meas_1_complex;

    %% compute the estimation of X0 and R2star
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

    % to avoid running into erros while doing parpool, just keep it running in the background and do not delete it
    %poolobj = gcp('nocreate');  % close parallel pool
    %delete(poolobj);
    warning('on','all')

    H_exp = exp(H);
    H_exp = min(H_exp, x0_max); 
    H_exp = max(H_exp, x0_min);


    R2_star = min(R2_star, r2star_max);
    R2_star = max(R2_star, r2star_min); 

    x0 = H_exp;
    r2star = R2_star;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% warm up the iterations %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ET1_mut_mat = zeros(sx,sy,sz,Nf);
    for (k=1:Nf)
        ET1_mut_mat(:,:,:,k) = (sin_FA_scaled(:,:,:,k).*(1-ET1))./(1-cos_FA_scaled(:,:,:,k).*ET1);
    end 

    R2_mut_mat  =zeros(sx,sy,sz,Ne);
    for (j=1:Ne)
        R2_mut_mat(:,:,:,j) = exp(-echo_time(j)*r2star);
    end 

    x_hat_all = x_mag_meas_1_complex;
    tau_x_hat_all = tau_x_mag_meas_1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% initialize with the previous spar prior recon %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    p_hat_psi = x_mag_meas_1_complex;
    tau_p_psi = tau_x_mag_meas_1;
    x_hat_all_tmp = x_mag_meas_1_complex;

    x_hat_psi = x_hat_meas_psi;

    % do an exponential fit of the tissue parameters
    x_mag_meas = abs(x_mag_meas_1_complex);

    for (iii=1:3)
            x0_new = zeros(sx,sy,sz);
            x0_new_denominator = zeros(sx,sy,sz);
            for (j=1:Ne)
                for (k=1:Nf)
                    T_tmp = R2_mut_mat(:,:,:,j) .* ET1_mut_mat(:,:,:,k);
                    x0_new = x0_new + T_tmp .* x_mag_meas(:,:,:,j,k);
                    x0_new_denominator = x0_new_denominator + T_tmp.^2;
                end
            end
            x0_new_denominator(x0_new_denominator==0) = eps;
            x0_tmp = x0_new ./ x0_new_denominator;


            x0_tmp = min(x0_tmp, x0_max);
            x0_tmp = max(x0_tmp, x0_min);
            x0_pre = x0;
            x0 = x0_tmp;

            cvg_x0 = norm(x0(:)-x0_pre(:), 'fro')/norm(x0(:), 'fro');

            r2star = exp_lsq_fit_fa_3d_r2_componentwise_dict_parfor(x_mag_meas, x0, ET1, r2_exp_dict, r2_exp_dict_sq_norm, r2_dict_val, echo_time, sin_FA_scaled, cos_FA_scaled, tan_FA_scaled, LN_num);
            r2star = min(r2star, r2star_max);
            r2star = max(r2star, r2star_min);


            ET1 = exp_lsq_fit_fa_3d_t1_componentwise_dict_parfor(x_mag_meas, x0, r2star, t1_exp_dict, echo_time,           sin_FA_scaled, cos_FA_scaled, tan_FA_scaled, LN_num);
            ET1 = min(ET1, 1-eps);
            ET1 = max(ET1, eps);

            ET1_mut_mat = zeros(sx,sy,sz,Nf);
            for (k=1:Nf)
                ET1_mut_mat(:,:,:,k) = (sin_FA_scaled(:,:,:,k).*(1-ET1))./(1-cos_FA_scaled(:,:,:,k).*ET1);
            end

            R2_mut_mat  =zeros(sx,sy,sz,Ne);
            for (j=1:Ne)
                R2_mut_mat(:,:,:,j) = exp(-echo_time(j)*r2star);
            end
    end

    %%% compute tau_x_hat_all and x_hat_all
    x_mag_exp = zeros(size(x_mag_meas));
    for (j=1:Ne)
        for (k=1:Nf)
            x_mag_exp(:,:,:,j,k) = x0 .* R2_mut_mat(:,:,:,j) .* ET1_mut_mat(:,:,:,k);
        end 
    end
    x_mag_exp = min(x_mag_exp, x_mag_max);
    x_mag_exp = max(x_mag_exp, x_mag_min);

    cvg_x0 = 1; 
    cvg_x_hat_all = 1;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% then move on to the exp reconstruction %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	for (ite_pe = 1:max_spar_mod_ite)

        %tic

		    % give fixed magnitude measurements
		    % should i let it run until convergence


            %%% iteration in the measurement block
            tau_p_meas_2 = A_2_echo_fa_comp_3d.multSq(tau_x_meas_psi);
            p_hat_meas_2 = A_2_echo_fa_comp_3d.mult(x_hat_meas_psi) - tau_p_meas_2 * s_hat_meas_2;

            fprintf('%d\n', mean(tau_p_meas_2(:)))

            for (ite_pe_est = 1:max_pe_est_ite)
                tau_w_2 = output_parameter_est(y, tau_w_2, p_hat_meas_2, tau_p_meas_2, kappa);
            end

            tau_s_meas_2 = 1 / (tau_w_2 + tau_p_meas_2);
            s_hat_meas_2 = (y - p_hat_meas_2) * tau_s_meas_2;

            tau_r_meas_2 = 1 / A_2_echo_fa_comp_3d.multSqTr(tau_s_meas_2);
            r_hat_meas_2 = x_hat_meas_psi + tau_r_meas_2 * A_2_echo_fa_comp_3d.multTr(s_hat_meas_2);

            for (ite_pe_est = 1:max_pe_est_ite)
                for (i=1:size(r_hat_meas_2,2))
                for (j=1:size(r_hat_meas_2,3))
                    lambda_x_hat_psi(i,j) = input_parameter_est(abs(r_hat_meas_2(:,i,j)), tau_r_meas_2, lambda_x_hat_psi(i,j), kappa);
                end
                end
            end

            x_hat_meas_psi_pre = x_hat_meas_psi;
            tau_x_meas_psi_tmp = zeros(size(r_hat_meas_2,2), size(r_hat_meas_2,3));
            for (i=1:size(r_hat_meas_2,2))
            for (j=1:size(r_hat_meas_2,3))
                [x_hat_meas_psi(:,i,j), tau_x_meas_psi_tmp(i,j)] = input_function(r_hat_meas_2(:,i,j), tau_r_meas_2, lambda_x_hat_psi(i,j));
            end
            end
            tau_x_meas_psi = mean(tau_x_meas_psi_tmp(:));
            %x_hat_psi = x_hat_psi_pre + damp_rate*(x_hat_psi-x_hat_psi_pre);

            %tau_x_hat_all_tmp = A_wav_fa_3d.multSq(tau_x_meas_psi);
            x_hat_all_tmp = A_wav_fa_3d.mult(x_hat_meas_psi);



            %%%%%%%%%
            %% estimate the parameters based on x_hat_all
            %%%%%%%%%
            for (ite_search = 1:max_search_ite)

            x_mag_meas_complex = x_hat_all_tmp;
            x_mag_meas = abs(x_mag_meas_complex);
            x_mag_meas(x_mag_meas<x_mag_min) = x_mag_min;


            x0_new = zeros(sx,sy,sz);
            x0_new_denominator = zeros(sx,sy,sz);
            for (j=1:Ne)
                for (k=1:Nf)
                    T_tmp = R2_mut_mat(:,:,:,j) .* ET1_mut_mat(:,:,:,k);
                    x0_new = x0_new + T_tmp .* x_mag_meas(:,:,:,j,k);
                    x0_new_denominator = x0_new_denominator + T_tmp.^2;
                end
            end
            x0_new_denominator(x0_new_denominator==0) = eps;
            x0_tmp = x0_new ./ x0_new_denominator;


            x0_tmp = min(x0_tmp, x0_max);
            x0_tmp = max(x0_tmp, x0_min);
            x0_pre = x0;
            x0 = x0_tmp;

            cvg_x0 = norm(x0(:)-x0_pre(:), 'fro')/norm(x0(:), 'fro');

            r2star = exp_lsq_fit_fa_3d_r2_componentwise_dict_parfor(x_mag_meas, x0, ET1, r2_exp_dict, r2_exp_dict_sq_norm, r2_dict_val, echo_time, sin_FA_scaled, cos_FA_scaled, tan_FA_scaled, LN_num);
            r2star = min(r2star, r2star_max);
            r2star = max(r2star, r2star_min);


            ET1 = exp_lsq_fit_fa_3d_t1_componentwise_dict_parfor(x_mag_meas, x0, r2star, t1_exp_dict, echo_time, sin_FA_scaled, cos_FA_scaled, tan_FA_scaled, LN_num);
            ET1 = min(ET1, 1-eps);
            ET1 = max(ET1, eps);

            ET1_mut_mat = zeros(sx,sy,sz,Nf);
            for (k=1:Nf)
                ET1_mut_mat(:,:,:,k) = (sin_FA_scaled(:,:,:,k).*(1-ET1))./(1-cos_FA_scaled(:,:,:,k).*ET1);
            end

            R2_mut_mat  =zeros(sx,sy,sz,Ne);
            for (j=1:Ne)
                R2_mut_mat(:,:,:,j) = exp(-echo_time(j)*r2star);
            end

            %%% compute tau_x_hat_all and x_hat_all
            x_mag_exp = zeros(size(x_mag_meas));
            for (j=1:Ne)
                for (k=1:Nf)
                    x_mag_exp(:,:,:,j,k) = x0 .* R2_mut_mat(:,:,:,j) .* ET1_mut_mat(:,:,:,k);
                end
            end
            x_mag_exp = min(x_mag_exp, x_mag_max);
            x_mag_exp = max(x_mag_exp, x_mag_min);

            end

            %%%% update the magnitude of x_hat_all with the fitted model parameters
            x_hat_all_pre = x_hat_all;
            x_hat_all = x_mag_exp .* sign(x_hat_all_tmp);

            x_hat_meas_psi_pre = x_hat_meas_psi;
            x_hat_meas_psi = A_wav_fa_3d.multTr(x_hat_all);

            % combine with r_hat_meas_2 and tau_r_meas_2

            cvg_x_hat_all = norm(x_hat_meas_psi(:)-x_hat_meas_psi_pre(:), 'fro')/norm(x_hat_meas_psi(:), 'fro');

		    cvg_gamp = max([cvg_x0 cvg_x_hat_all]);
		    if ((cvg_gamp<cvg_thd) && (ite_gamp>2))
		    	break;
		    end

            fprintf('Ite %d CVG PE: %d\t%d\n', ite_pe, cvg_x0, cvg_x_hat_all)
		    

        %toc

        tau_w_2

        %fprintf('Ite %d CVG PE: %d\t%d\n', ite_pe, cvg_x0, cvg_x_hat_all)

        cvg_pe = max([ cvg_x0 cvg_x_hat_all ]);
        if ((cvg_pe<cvg_thd)&&(ite_pe>2))
            break;
        end
	end
	
    res.x0 = x0;
    res.r2star = r2star;
    res.ET1 = ET1;

    res.x_hat_all = x_hat_all;
    res.tau_x_hat_all = tau_x_hat_all;
    res.s_hat_meas_2 = s_hat_meas_2;


    input_par.lambda_x_hat_psi = lambda_x_hat_psi;
    output_par.tau_w_2 = tau_w_2;

end

function [x0_hat, tau_x0] = input_function(r_hat, tau_r, lambda)

    thresh = lambda * tau_r;
    x0_hat = max(0, abs(r_hat)-thresh) .* sign(r_hat);

    tau_x0 = tau_r;
    %tau_x0(abs(x0_hat)==0) = 0;
    tau_x0 = tau_r * length(x0_hat(abs(x0_hat)>0)) / length(x0_hat);

end


function lambda = input_parameter_est(r_hat, tau_r, lambda, kappa)

    lambda_pre = lambda;

    % do we need some kind of normalization here?
    dim_smp=length(r_hat);
    num_cluster=1;

    block_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        block_mat(:,i) = 0.5/tau_r * (tau_r*lambda(i)-r_hat).^2;
    end

    block_mat_min = [];
    if (num_cluster==1)
        block_mat_min = block_mat;
    else
        block_mat_min = max(block_mat')';
    end

    block_mat = block_mat - block_mat_min; % subtract the minimum value of each row
    block_mat_two_exp = exp(block_mat);
    block_mat_one_exp = exp(-block_mat_min);

    block_mat_erfc = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        block_mat_erfc(:,i) = erfc(sqrt(0.5/tau_r)*(tau_r*lambda(i)-r_hat));
    end

    lambda_tmp_mat_0 = zeros(dim_smp, num_cluster);
    for (i = 1:num_cluster)
        lambda_tmp_mat_0(:,i) = lambda(i)/2 * block_mat_two_exp(:,i) .* block_mat_erfc(:,i);
    end

    % compute lambda
    % compute the first order derivative

    der_block = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        der_block(:,i) = ( sqrt(2*tau_r/pi) ./ erfcx(sqrt(0.5/tau_r)*(tau_r*lambda(i)-r_hat)) + r_hat - tau_r*lambda(i) );
    end

    fst_der_lambda = zeros(dim_smp, num_cluster);
    for (i = 1:num_cluster)
        fst_der_lambda(:,i) = 1/lambda(i) - der_block(:,i);
    end

    scd_der_lambda = zeros(dim_smp, num_cluster);
    for (i = 1:num_cluster)
        scd_der_lambda(:,i) = -1/lambda(i)/lambda(i) + ( tau_r + (r_hat - tau_r*lambda(i)).*der_block(:,i) ) - der_block(:,i).^2;
    end

    lambda_tmp_mat = lambda_tmp_mat_0;
    lambda_tmp_mat_sum = sum(lambda_tmp_mat, 2);
    for (i=1:num_cluster)
        lambda_tmp_mat(:,i) = lambda_tmp_mat(:,i) ./ ( lambda_tmp_mat_sum + eps);
    end

    lambda_tmp_mat_1 = lambda_tmp_mat;
    lambda_tmp_mat_2 = lambda_tmp_mat;
    %for (i=1:num_cluster)
    %    lambda_tmp_mat_1(:,i) = lambda_tmp_mat_1(:,i) .* (fst_der_lambda(:,i) - scd_der_lambda(:,i)*lambda(i));
    %    lambda_tmp_mat_2(:,i) = lambda_tmp_mat_2(:,i) .* scd_der_lambda(:,i);
    %end
    %lambda_new = - sum(lambda_tmp_mat_1) ./ (sum(lambda_tmp_mat_2) + eps);   % to avoid division by 0
    %lambda_new = lambda_new';


    lambda_new = [];
    for (i=1:num_cluster)
        lambda_tmp_mat_1_tmp = lambda_tmp_mat_1(:,i) .* fst_der_lambda(:,i);
        lambda_tmp_mat_2_tmp = lambda_tmp_mat_2(:,i) .* scd_der_lambda(:,i);
        lambda_new_tmp = 0;
        if (sum(lambda_tmp_mat_2_tmp)<0)
            lambda_new_tmp = lambda(i) - sum(lambda_tmp_mat_1_tmp)/sum(lambda_tmp_mat_2_tmp);
        else
            if (sum(lambda_tmp_mat_1_tmp)>0)
                lambda_new_tmp = lambda(i)*1.1;
            else
                lambda_new_tmp = lambda(i)*0.9;
            end
        end
        lambda_new = [lambda_new; lambda_new_tmp];
    end

    lambda_new = max(lambda_new, 1e-12);  % necessary to avoid 0 which leads to NaN

    % lambda_new could be negative, causing problem
    lambda = lambda + kappa * (lambda_new - lambda);

end



function tau_w = output_parameter_est(y, tau_w, p_hat, tau_p, kappa)

    %tau_w_new = max(mean( abs(p_hat(:)-y(:)).^2 - tau_p(:) ), eps);   % make sure the variance is non-negative
    tau_w_new = mean( abs(p_hat(:)-y(:)).^2 ) + tau_p;
    %tau_w_new = mean( (p_hat-y).^2 - tau_p );
    %if (tau_w_new<0)
    %    tau_w_new = tau_w;
    %end

    tau_w = tau_w + kappa * (tau_w_new - tau_w);
end

