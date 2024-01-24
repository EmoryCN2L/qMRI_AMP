 function V=exponential_fit(x_mag, x0, r2star, t1_exp_dict, echo_time, sin_FA_scaled, cos_FA_scaled, tan_FA_scaled, LN_num )
 
    % solve the least square problem for R2 only while keeping x0 fixed

    % x_mag        : the recovered magnitude echo images of size sx by sy by Ne
    %           : sx is the resolution in the x direction
    %           : sy is the resolution in the y direction
    %           : Ne is the number of echoes
 
    % par       : various parameters
 
    % Assigning parameres according to par
 
    % maximum and minimum value of Rt_abs
 

    %parpool(LN_num)

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

    x0_process = zeros([sx sy sz Ne]);
    for (i=1:Ne)
        x0_process(:,:,:,i) = x0.*exp(-echo_time(i)*r2star);
    end
    x0_process_sq_norm = squeeze(sum(x0_process.^2,4));
 
    x_mag_sq_norm = squeeze(sum(x_mag.^2,4));   % sx sy sz Nf
    x_mag_sq_norm = squeeze(sum(x_mag_sq_norm,4));  % sx sy sz
    
    min_distance_to_dict_all = zeros(sx,sy,sz,LN_num);
    V_all = zeros(sx,sy,sz,LN_num);

    t1_exp_dict_len = length(t1_exp_dict);
    t1_exp_dict_part = floor(t1_exp_dict_len/LN_num);

    parfor (l_idx = 1:LN_num)

    k_seq = ((l_idx-1)*t1_exp_dict_part + 1) : (l_idx * t1_exp_dict_part);

    if (l_idx == LN_num)
        k_seq = ((l_idx-1)*t1_exp_dict_part + 1) : t1_exp_dict_len;
    end

    min_distance_to_dict_part = zeros(sx,sy,sz);
    V_part = zeros(sx,sy,sz);

    for (k = k_seq)
        %if (mod(k,500)==0)
        %fprintf('%d\n', k)
        %end
        t1_exp_dict_seq = t1_exp_dict(k);
        t1_mut_mat = zeros(sx,sy,sz,Nf);
        for (j=1:Nf)
            t1_mut_mat(:,:,:,j) = (sin_FA_scaled(:,:,:,j)*(1-t1_exp_dict_seq))./(1-cos_FA_scaled(:,:,:,j)*t1_exp_dict_seq);
        end
        t1_mut_sq = squeeze(sum(t1_mut_mat.^2,4));

        distance_to_dict = zeros(sx,sy,sz);

        for (i=1:Ne)
            distance_to_dict_tmp = zeros(sx,sy,sz);
            for (j=1:Nf)
                distance_to_dict_tmp = distance_to_dict_tmp + x_mag(:,:,:,i,j).*t1_mut_mat(:,:,:,j);
            end
            distance_to_dict = distance_to_dict + x0_process(:,:,:,i).*distance_to_dict_tmp;
        end
        distance_to_dict = x_mag_sq_norm + x0_process_sq_norm.*t1_mut_sq - 2*distance_to_dict;
        
        if (k==k_seq(1))
            V_part = repmat(t1_exp_dict(k),[sx sy sz]);
            min_distance_to_dict_part = distance_to_dict;
        else
            V_part(min_distance_to_dict_part>distance_to_dict) = t1_exp_dict(k);
            min_distance_to_dict_part = min(min_distance_to_dict_part, distance_to_dict);
        end            
    end

    V_all(:,:,:,l_idx) = V_part;
    min_distance_to_dict_all(:,:,:,l_idx) = min_distance_to_dict_part;

    end

    min_distance_to_dict = min_distance_to_dict_all(:,:,:,1);
    V = V_all(:,:,:,1);

    for (l_idx=2:LN_num)
        V_tmp = V_all(:,:,:,l_idx);
        V(min_distance_to_dict>min_distance_to_dict_all(:,:,:,l_idx)) = V_tmp(min_distance_to_dict>min_distance_to_dict_all(:,:,:,l_idx));
        min_distance_to_dict = min(min_distance_to_dict, min_distance_to_dict_all(:,:,:,l_idx));
    end

    %poolobj = gcp('nocreate');  % close parallel pool
    %delete(poolobj);

end
