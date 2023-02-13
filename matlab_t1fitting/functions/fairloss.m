function L = fairloss(r, H, g)
    r = r(:); 
    if nargin == 2
        g = 0; 
    end
    H = sqrt(H);
    rabs = abs(r);
    if g == 0
        L = 2 * H * H * ( rabs / H  - log(1 + rabs / H) );
    else
        error("Not implemented!");
    end


end