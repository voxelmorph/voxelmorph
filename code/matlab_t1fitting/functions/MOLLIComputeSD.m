function SD = MOLLIComputeSD(p, tvec, dvec)
    
    pred = MOLLI3param(p, tvec);
    r = abs(pred(:) - dvec(:));
    r = sort(r);
    s = median( r(3:end) ) * 1.48; % sigma estimation
    
    % compute D1/2
%     A = p(1); B = p(1) * p(2); 
    T1 = p(3) * ( p(2) - 1);
    D = zeros( 3, 3, length(tvec) );
%     dA = 1 - B * exp( - tvec * (p(2) - 1) / T1 ) .* tvec * B / T1 / A /A;
%     dB = -exp(- tvec *( p(2) - 1 ) / T1 ) + B*exp(- tvec *( p(2) - 1) / T1 ) .* tvec / T1 /A;
%     dT1 = -B * exp( -tvec * (p(2) - 1)/ T1) .* tvec * (p(2) - 1) / T1 / T1;
    c = p(1); k = p(2);
    dc = 1 - k * exp( - tvec * (k - 1) / T1);
    dk = -c * exp(-tvec * (k - 1 ) / T1) + c * k / T1 * tvec .* exp( - tvec * (k -1)/T1);
    dT1 = - c * k * exp( -tvec * (k -1) / T1) .* tvec * (k-1) / T1 / T1;
    for j = 1:1:length(tvec)
       D(:, :, j) = [dT1(j); dc(j); dk(j)] * [dT1(j), dc(j), dk(j)]; 
    end
    D = squeeze( sum(D, 3) ) / s / s;
    Dinv = pinv(D + 1e-5 * eye(3) );
    SD = sqrt(Dinv(1, 1));
    
end