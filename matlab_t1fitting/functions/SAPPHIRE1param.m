function s = SAPPHIRE1param(p, TI, TD, M0)
%% SAPPHIRE2param.m
%   The function computes the 2-parameter SAPPHIRE model.
        funcS1 = @(t,T1) (1 - exp(-t ./ T1) );
        fun = @(x,xd) x(1) .* (1 - (1 + funcS1( max(TD(:) - xd(:),0) ,x(3))) .* exp(-xd(:) ./ x(3)) );
        s = fun([M0 2 p], TI);
end