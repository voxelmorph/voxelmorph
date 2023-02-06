function s = SAPPHIRE2param(p, TI, TD)
%% SAPPHIRE2param.m
%   The function computes the 2-parameter SAPPHIRE model.
        funcS1 = @(t,T1) (1 - exp(-t ./ T1) );
        fun = @(x,xd) x(1) .* (1 - (1 + funcS1( max(TD(:) - xd(:),0) ,x(3))) .* exp(-xd(:) ./ x(3)) );
        s = fun([p(1) 2 p(2)], TI);
end