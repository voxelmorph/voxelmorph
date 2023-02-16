function L = welschloss(r, H, g)
    r = r(:); 
    if nargin == 2
        g = 0; 
    end
    if g == 0
        L =  H * ( 1 - exp(- r .* r /H) );
    else
        error("Not implemented!");
    end


end