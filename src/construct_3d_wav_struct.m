% construct linear measurement operator and its transpose operator
function wav_struct = construct_3d_wav_struct(wav_coef, wav_vessel)
    for (i=1:length(wav_vessel.dec))
        wav_vessel_tmp = wav_vessel.dec{i};
        wav_coef_tmp_len = length(wav_vessel_tmp(:));
        wav_vessel_tmp = reshape(wav_coef(1:wav_coef_tmp_len), size(wav_vessel_tmp));
        wav_vessel.dec{i} = wav_vessel_tmp;
        wav_coef(1:wav_coef_tmp_len)=[];
    end
    wav_struct = wav_vessel;
end

