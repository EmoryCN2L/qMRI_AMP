% construct linear measurement operator and its transpose operator
function y = A_op_3d_cylinder_combine(X, s_vect)
    X_fft = fft(X,[],1);
    X_fft = fft(X_fft,[],2);
    X_fft = fft(X_fft,[],3);

    X_fft = fftshift(X_fft, 1);    % it seems that complex images do not need fftshift???
    X_fft = fftshift(X_fft, 2);
    X_fft = fftshift(X_fft, 3);

    X_fft = permute(X_fft, [2 3 1 4]);
    X_fft = reshape(X_fft, [size(X_fft,1)*size(X_fft,2) size(X_fft,3) size(X_fft,4)]);
    y = X_fft(s_vect,:,:);
    y = permute(y,[2 1 3]);
end

