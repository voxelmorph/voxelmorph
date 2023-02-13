function [SD, s] = SAPPHIREComputeSD(p, TI, TD, dvec)

    % sigma-estimation
    pred = SAPPHIRE2param(p, TI, TD);
    r = abs( pred(:) - dvec(:) );
    r = sort(r);
    s = median(r(2:end)) * 1.48;
    
    % compute D1/2
    M0 = p(1); T1 = p(2);
    dM0 = 1 - 2 * exp(-TI / T1) + exp(-TD/T1);
    Td = TD - TI; 
    dT1 = -M0 * exp( - TI/ T1) .* ( 2 * TI / T1/T1 - Td / T1/ T1 .* exp( - Td / T1) - exp(-Td / T1) .* TI / T1 / T1);
%     dT1 =  (SAPPHIRE2param(p + [0, 1e-1], TI, TD) - SAPPHIRE2param(p , TI, TD)) * 1e1;
%     dM0 =  (SAPPHIRE2param(p + [1e-4, 0], TI, TD) - SAPPHIRE2param(p , TI, TD)) * 1e4;
    dM0(TI > 9999) = 1;
    dT1(TI > 9999) = 1e-8;
    D = zeros(2, 2, length(TI));
    for j = 1:1:length(TI)
        D(:, :, j) = [dT1(j); dM0(j)] * [dT1(j), dM0(j)];
    end
    D = sum(D, 3) / s / s;
    Dinv = inv(D + 1e-6 * eye(2));
    SD = sqrt(Dinv(1, 1));
end