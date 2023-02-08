function parsave(fname, fd)
    data = fd{1};
    pmap_mse = fd{2};
    sd_mse = fd{3};
    contour = fd{4};
    null_index = fd{5};
    S = fd{6};
    areamask = fd{7};
    epi_BW = fd{8};
    endo_BW = fd{9};
    save(fname, ...
        'data', 'pmap_mse', 'sd_mse', 'contour', 'null_index', 'S', 'areamask', 'epi_BW', 'endo_BW');
end