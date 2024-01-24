function [res, input_par, output_par] = gamp_mri_full_x0_r2star_l1_3d_nw_com_com_spar_ahead(A_2_echo_fa_comp_3d, y, gamp_par, input_par, output_par)

    % no wavelet in A_2_echo_nw_3d
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
    max_spar_ite = gamp_par.max_spar_ite;
    max_pe_est_ite = gamp_par.max_pe_est_ite;
	cvg_thd = gamp_par.cvg_thd;
	kappa = gamp_par.kappa;

    tau_x_meas_psi = gamp_par.tau_x_meas_psi;
    x_hat_meas_psi = gamp_par.x_hat_meas_psi;
    s_hat_meas_1 = gamp_par.s_hat_meas_1;

    % set input distribution parameters
    lambda_x_hat_psi = input_par.lambda_x_hat_psi;

    tau_w_1 = output_par.tau_w_1;

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

    damp_rate = gamp_par.damp_rate;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% initialize tau_w_exp with tau_x_mag_meas from the warm up step %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% first finish the sparse reconstruction %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for (ite_pe = 1:max_spar_ite)


            tau_p_meas_1 = A_2_echo_fa_comp_3d.multSq(tau_x_meas_psi);
            p_hat_meas_1 = A_2_echo_fa_comp_3d.mult(x_hat_meas_psi) - tau_p_meas_1 * s_hat_meas_1;

            fprintf('%d\n', mean(tau_p_meas_1(:)))
            % parameter estimation
            for (ite_pe_est = 1:max_pe_est_ite)
                tau_w_1 = output_parameter_est(y, tau_w_1, p_hat_meas_1, tau_p_meas_1, kappa);
            end

            tau_s_meas_1 = 1 / (tau_w_1 + tau_p_meas_1);
            s_hat_meas_1 = (y - p_hat_meas_1) * tau_s_meas_1;

            %%%%% the first sparse info from multi echo images
            tau_r_meas_1 = 1 / A_2_echo_fa_comp_3d.multSqTr(tau_s_meas_1);
            r_hat_meas_1 = x_hat_meas_psi + tau_r_meas_1 * A_2_echo_fa_comp_3d.multTr(s_hat_meas_1);

            % parameter estimation
            for (ite_pe_est = 1:max_pe_est_ite)
                for (i=1:size(r_hat_meas_1,2))
                for (j=1:size(r_hat_meas_1,3))
                    lambda_x_hat_psi(i,j) = input_parameter_est(abs(r_hat_meas_1(:,i,j)), tau_r_meas_1, lambda_x_hat_psi(i,j), kappa);
                end
                end
            end

            x_hat_meas_psi_pre = x_hat_meas_psi;
            tau_x_meas_psi_tmp = zeros(size(r_hat_meas_1,2), size(r_hat_meas_1,3));
            for (i=1:size(r_hat_meas_1,2))
            for (j=1:size(r_hat_meas_1,3))
                [x_hat_meas_psi(:,i,j), tau_x_meas_psi_tmp(i,j)] = input_function(r_hat_meas_1(:,i,j), tau_r_meas_1, lambda_x_hat_psi(i,j));
            end
            end
            tau_x_meas_psi = mean(tau_x_meas_psi_tmp(:));
            x_hat_meas_psi = x_hat_meas_psi_pre + damp_rate*(x_hat_meas_psi-x_hat_meas_psi_pre);

            cvg_gamp_x_hat_meas_psi = norm(x_hat_meas_psi(:)-x_hat_meas_psi_pre(:), 'fro')/norm(x_hat_meas_psi(:), 'fro');

            cvg_gamp = max([ cvg_gamp_x_hat_meas_psi]); % max([cvg_gamp_x_hat_meas_psi cvg_gamp_x_mag_exp]);
            if ((cvg_gamp<cvg_thd) && (ite_gamp>2))
                break;
            end

        lambda_x_hat_psi
        tau_w_1

        fprintf('Ite %d CVG PE: %d\n', ite_pe, cvg_gamp_x_hat_meas_psi)

        cvg_pe = max([ cvg_gamp_x_hat_meas_psi]);
        if ((cvg_pe<cvg_thd)&&(ite_pe>2))
            break;
        end

    end

    res.x_hat_meas_psi = x_hat_meas_psi;
    res.tau_x_meas_psi = tau_x_meas_psi;
    res.s_hat_meas_1 = s_hat_meas_1;

    input_par.lambda_x_hat_psi = lambda_x_hat_psi;
    output_par.tau_w_1 = tau_w_1;

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

