function X = At_op_3d_cylinder_combine(maps_conj, y, s_vect)   % assume square region with a odd-number size

    y = permute(y, [2 1 3]);

    X_tmp = gpuArray(zeros([size(maps_conj,2)*size(maps_conj,3) size(maps_conj,1) size(maps_conj,4)]));
    X_tmp(s_vect,:,:) = y;
    X_tmp = reshape(X_tmp, [size(maps_conj,2) size(maps_conj,3) size(maps_conj,1) size(maps_conj,4)]);
    X_tmp = permute(X_tmp, [3 1 2 4]);
    
    X_tmp = fftshift(X_tmp,1);
    X_tmp = fftshift(X_tmp,2);
    X_tmp = fftshift(X_tmp,3);

    X_tmp = ifft(X_tmp,[],1);
    X_tmp = ifft(X_tmp,[],2);
    X_tmp = ifft(X_tmp,[],3);

    X = sum(X_tmp.*maps_conj, 4);

end

