function L = huberloss(r, H, g)
%% huberloss.m
%   The function compute the g-th order gradient for Huber loss. Huber(r) =
%   0.5*r^2 for |r| < sqrt(H), if |r| > sqrt(H), Huber(r) = sqrt(H) *
%   abs(r) - 0.5 * H.
%   If g == 0, the function returns the Huber loss itself. H is the Huber loss
%   hyperparameter. r is an array of residuals.
    r = r(:); 
    if nargin == 2
        g = 0; 
    end
    H = sqrt(H) ;
    if g == 0
        L = 0.5 .* r .* r;
        L ( abs(r) > H ) = H .* abs( r(abs(r) > H) ) - 0.5 * H * H;  
    elseif g == 1
        L = r ;
        L ( abs(r) > H ) = H .* sign(r) ;  
    elseif g == 2
        L = 1;
        L ( abs(r) > H ) = 0 ; 
    else
        error("g must be 0, 1, or 2, input: %d", g);
    end
end