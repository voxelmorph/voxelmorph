function L = cauchyloss(r, H, g)
%% cauchyloss.m
%   This function computes cauchy loss of residuals r, given hyperparameter
%   H. If g is given, the function computes g-th order gradient of
%   cauchyloss, where g = 0, 1, or 2. Cauchy(r) = H * log( 1 + r^2 / H).
    if nargin == 2
        g = 0;
    end
    if g == 0
        e = r .* r;
        L = H * log(1+e/H);
    else
        error("Not implemented yet!");
    end
end