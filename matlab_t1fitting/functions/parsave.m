function parsave(fname, fd)
    data = fd{1};
    pmap_mse = fd{2};
    pmap_huber = fd{3};
    pmap_fair = fd{4};
    pmap_cauchy = fd{5};
    pmap_welsch = fd{6};
    sd_mse = fd{7};
    sd_huber = fd{8};
    sd_fair = fd{9};
    sd_cauchy = fd{10};
    sd_welsch = fd{11};
    contour = fd{12};
    null_index = fd{13};
    S = fd{14};
    save(fname, ...
        'data', 'pmap_huber', 'sd_huber', ...
        'pmap_cauchy', 'sd_cauchy', ...
        'pmap_fair', 'sd_fair', ...
        'pmap_welsch', 'sd_welsch', ...
        'pmap_mse', 'sd_mse', 'contour', 'null_index', 'S');
end