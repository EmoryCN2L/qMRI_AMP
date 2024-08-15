function V=exp_lsq_fit_fa_3d_r2_componentwise_dict(x_mag, x0, ET1, exp_dict, exp_dict_sq_norm, r2_dict_val, echo_time, sin_FA_scaled, cos_FA_scaled, tan_FA_scaled)
 
    % solve the least square problem for R2 only while keeping x0 fixed

    % x_mag        : the recovered magnitude echo images of size sx by sy by Ne
    %           : sx is the resolution in the x direction
    %           : sy is the resolution in the y direction
    %           : Ne is the number of echoes
 
    % par       : various parameters
 
    % Assigning parameres according to par
 
    % maximum and minimum value of Rt_abs

    %parpool(LN_num)

    % convert all the inputs to gpuArrays
    x_mag = gpuArray(x_mag);
    x0 = gpuArray(x0);
    ET1 = gpuArray(ET1);
    exp_dict = gpuArray(exp_dict);
    exp_dict_sq_norm = gpuArray(exp_dict_sq_norm);
    r2_dict_val = gpuArray(r2_dict_val);
    echo_time = gpuArray(echo_time);
    sin_FA_scaled = gpuArray(sin_FA_scaled);
    cos_FA_scaled = gpuArray(cos_FA_scaled);
    tan_FA_scaled = gpuArray(tan_FA_scaled);
 
    sx = size(x_mag,1);
    sy = size(x_mag,2);
    sz = size(x_mag,3);

    Ne = size(x_mag,4);
    Nf = size(x_mag,5);
 
    x0_min = eps;
    x_mag_min = eps;
    V_min = 0;  % or eps???
 
    % initialize x0 and V
    x_mag = max(x_mag, x_mag_min);
     
    % initialize H, x0, and V
    x0 = max(x0, x0_min);

    x0_process = gpuArray(zeros([sx sy sz Nf]));
    for (j=1:Nf)
        x0_process(:,:,:,j) = x0.*sin_FA_scaled(:,:,:,j).*(1-ET1)./(1-cos_FA_scaled(:,:,:,j).*ET1);
    end
    x0_process_sq_norm = squeeze(sum(x0_process.^2,4));
 
    x_mag_sq_norm = squeeze(sum(x_mag.^2,4));   % sx sy sz Nf
    x_mag_sq_norm = squeeze(sum(x_mag_sq_norm,4));  % sx sy sz
    

    exp_dict_len = length(exp_dict);

    k_seq = 1:exp_dict_len;

    zero_gpu_temp = gpuArray(zeros(sx,sy,sz));
    min_distance_to_dict_part = zero_gpu_temp;
    V_part = zero_gpu_temp;

    for (k=1:exp_dict_len)
        %if (mod(k,500)==0)
        %fprintf('%d\n', k)
        %end
        exp_dict_seq = exp_dict(:,k);
        distance_to_dict = zero_gpu_temp;
        for (j=1:Nf)
            distance_to_dict_tmp = zero_gpu_temp;
            for (i=1:Ne)
                distance_to_dict_tmp = distance_to_dict_tmp + x_mag(:,:,:,i,j)*exp_dict_seq(i);
            end
            distance_to_dict = distance_to_dict + x0_process(:,:,:,j).*distance_to_dict_tmp;
        end
        distance_to_dict = x_mag_sq_norm + x0_process_sq_norm*exp_dict_sq_norm(k) - 2*distance_to_dict;
        
        if (k==1)
            V_part = gpuArray(repmat(r2_dict_val(k),[sx sy sz]));
            min_distance_to_dict_part = distance_to_dict;
        else
            V_part(min_distance_to_dict_part>distance_to_dict) = r2_dict_val(k);
            min_distance_to_dict_part = min(min_distance_to_dict_part, distance_to_dict);
        end
    end

    V = gather(V_part);
    g_device = gpuDevice;
    reset(g_device);

end
