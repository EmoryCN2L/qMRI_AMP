% construct linear measurement operator and its transpose operator
function y = A_op_3d_cylinder(X, s_vect)
    X_fft = fftshift(fftn(X));    % it seems that complex images do not need fftshift???
    y = zeros(size(X,1), length(s_vect));
    for (i=1:size(X,1))
        X_fft_tmp = squeeze(X_fft(i,:,:));
        y(i,:) = X_fft_tmp(s_vect);
    end
end

