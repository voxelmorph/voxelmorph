function [myo, inendo, inepi] = get_myomap(h, w, contour)
%% get_myomap.m
    [X, Y] = meshgrid(1:1:h, 1:1:w);
    endo = contour.endo;
    epi = contour.epi;
    inendo = inpolygon( X(:), Y(:), endo(:, 1), endo(:, 2) );
    inepi = inpolygon( X(:), Y(:), epi(:, 1), epi(:, 2) );
    inendo = reshape(inendo, [h, w]);
    inepi = reshape(inepi, [h, w]);
    myo = inepi .* ( 1 - inendo );
end