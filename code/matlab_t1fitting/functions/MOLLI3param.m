function S = MOLLI3param(p, t)
%% MOLLI3param.m 
%   S = MOLLI3param(p, t), 3-parameter MOLLI T1 mapping model
%   S = p(1) * ( 1 - p(2) * exp( -t / p(3) );
    t = t(:) ; % ensure column vector
    S = p(1) * ( 1 - p(2) * exp( -t / p(3) ) );
end