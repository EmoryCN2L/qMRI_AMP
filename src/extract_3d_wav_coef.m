% extract 3d wavelet_coefficients
function wav_coef = extract_3d_wav_coef(wav_struct)
    wav_coef=[];
    for (i=1:length(wav_struct.dec))
        wav_mat_tmp = wav_struct.dec{i};
        wav_coef = [wav_coef; wav_mat_tmp(:)];
    end
end

