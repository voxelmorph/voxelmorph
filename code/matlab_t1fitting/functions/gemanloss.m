function L = gemanloss(r, H, g)
    r = r(:); 
    if nargin == 2
        g = 0; 
    end
    if g == 0
        L = H * r .* r / ( H + r .* r);
    else
        error("Not implemented!");
    end

end