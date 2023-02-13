function [SD, s] = SAPPHIREComputeSD2(p, TI, TD, dvec)

    % sigma-estimation
    pred = SAPPHIRE1param(p(2), TI, TD, dvec(end) );
    r = abs( pred(:) - dvec(:) );
    r = sort(r);
    s = median(r(1:end)) * 1.48;
    
    % compute D1/2
    T1 = p(2);
    Td = TD - TI; 
    M0 = p(1);
    dT1 = -M0 * exp( - TI/ T1) .* ( 2 * TI / T1/T1 - Td / T1/ T1 .* exp( - Td / T1) - exp(-Td / T1) .* TI / T1 / T1);
    dT1(TI > 9999) = 0;
    D = sum( dT1 .* dT1, 'all') / s / s;
    Dinv = 1/(D + 1e-6 );
    SD = sqrt(Dinv);
end