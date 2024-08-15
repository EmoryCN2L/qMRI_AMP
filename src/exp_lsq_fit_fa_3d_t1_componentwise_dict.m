function V=exp_lsq_fit_fa_3d_t1_componentwise_dict(x_mag, x0, r2star, t1_exp_dict, echo_time, sin_FA_scaled, cos_FA_scaled, tan_FA_scaled)
 
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
    r2star = gpuArray(r2star);
    t1_exp_dict = gpuArray(t1_exp_dict);
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
 
    % initialize x0 and V
    x_mag = max(x_mag, x_mag_min);
     
    % initialize H, x0, and V
    x0 = max(x0, x0_min);

    x0_process = gpuArray(zeros([sx sy sz Ne]));
    for (i=1:Ne)
        x0_process(:,:,:,i) = x0.*exp(-echo_time(i)*r2star);
    end
    x0_process_sq_norm = squeeze(sum(x0_process.^2,4));
 
    x_mag_sq_norm = squeeze(sum(x_mag.^2,4));   % sx sy sz Nf
    x_mag_sq_norm = squeeze(sum(x_mag_sq_norm,4));  % sx sy sz
    

    t1_exp_dict_len = length(t1_exp_dict);

    zero_gpu_temp = gpuArray(zeros(sx,sy,sz));
    min_distance_to_dict_part = zero_gpu_temp;
    V_part = zero_gpu_temp;

    % permutation for a more efficient implementation
    x_mag = permute(x_mag, [1 2 3 5 4]);

    for (k = 1:t1_exp_dict_len)
        %if (mod(k,500)==0)
        %fprintf('%d\n', k)
        %end
        t1_exp_dict_seq = t1_exp_dict(k);
        %t1_mut_mat = zeros(sx,sy,sz,Nf);
        %for (j=1:Nf)
        %    t1_mut_mat(:,:,:,j) = (sin_FA_scaled(:,:,:,j)*(1-t1_exp_dict_seq))./(1-cos_FA_scaled(:,:,:,j)*t1_exp_dict_seq);
        %end

        t1_mut_mat = (sin_FA_scaled*(1-t1_exp_dict_seq))./(1-cos_FA_scaled*t1_exp_dict_seq);
        t1_mut_sq = squeeze(sum(t1_mut_mat.^2,4));

        distance_to_dict = zero_gpu_temp;

        for (i=1:Ne)

            % the following is the implementation when we do not permute x_mag 
            %distance_to_dict_tmp = zeros(sx,sy,sz);
            %for (j=1:Nf)
            %    distance_to_dict_tmp = distance_to_dict_tmp + x_mag(:,:,:,i,j).*t1_mut_mat(:,:,:,j);
            %end

            distance_to_dict_tmp = sum(x_mag(:,:,:,:,i).*t1_mut_mat, 4);
            distance_to_dict = distance_to_dict + x0_process(:,:,:,i).*distance_to_dict_tmp;
        end
        distance_to_dict = x_mag_sq_norm + x0_process_sq_norm.*t1_mut_sq - 2*distance_to_dict;
        
        if (k==1)
            V_part = gpuArray(repmat(t1_exp_dict(k),[sx sy sz]));
            min_distance_to_dict_part = distance_to_dict;
        else
            V_part(min_distance_to_dict_part>distance_to_dict) = t1_exp_dict(k);
            min_distance_to_dict_part = min(min_distance_to_dict_part, distance_to_dict);
        end            
    end

    V = gather(V_part);
    
    g_device = gpuDevice;
    reset(g_device);

end
