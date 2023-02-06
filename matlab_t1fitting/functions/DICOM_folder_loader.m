function data = DICOM_folder_loader(folder)
    dcmfiles = dir( strcat( folder, '/*.dcm') );
    num_images = length(dcmfiles);
    
    % the file name of an dicom image follows this pattern:
    %   "[PREFIX]_[RECONTYPE]_IM_[IMAGEINDEX].dcm"
    prefices = zeros(num_images, 1); 
    recon_type = cell(num_images, 1); 
    image_indices = zeros(num_images, 1);
    
    for file_no = 1:1:num_images
        file_struct = dcmfiles( file_no ); 
        file_name = file_struct.name;
        file_name_parts = split( file_name, '_');
        % parse file name
        prefices(file_no) = int64( str2double( file_name_parts{1} ) );  
        recon_type{file_no} = file_name_parts{2};
        image_indices(file_no) = int64( str2double( file_name_parts{4} (1:end-4) ) ); 
    end
    
    mappings = unique(prefices, 'sorted' ) ;
    data = cell(length(mappings), 1);
    
    for nm = 1:1:length( mappings )
        m = mappings(nm);
        
        % find images files of this mapping series
        image_file_nums = find(prefices == m);
        image_indices_m = image_indices(prefices == m);
        [~, image_indices_mi] = sort( image_indices_m ); % sort by image index
        image_file_nums = image_file_nums( image_indices_mi );
        
        % read first image frame for image dimension, inversion times and recontype
        file_struct = dcmfiles(image_file_nums(1)); 
        X = double( dicomread( strcat( file_struct.folder, '/',  file_struct.name) ) );
        dcmheader = dicominfo( strcat( file_struct.folder, '/',  file_struct.name));
        spacing = dcmheader.PixelSpacing;
        data_mat_m = zeros( [size(X), length(image_file_nums)] ); % data matrix
%         fprintf("Loading file %s \n", file_struct.name); % for debugging
        try
            wip_m = readinSiemensIMAData( strcat( file_struct.folder, '/', file_struct.name), 1); % TI
            recontype_m = recon_type{ image_file_nums(1) };
            data_mat_m(:, :, 1) = X ;
            % read the rest frames
            for j = 2:1:length(image_file_nums)
                file_struct = dcmfiles(image_file_nums(j)); 
                X = dicomread( strcat( file_struct.folder, '/', file_struct.name) );
                data_mat_m (:, :, j) = X ;
            end
        catch
            data_m.image_num = -1;
            data_m.data_mat = 0;
            data_m.recontype = '';
            data_m.wip = [];
            data_m.n_images = -1 ;
            data_m.spacing = spacing;
            data{nm} = data_m;
            continue;
        end
        % data structure
        data_m.image_num = m;
        data_m.data_mat = data_mat_m;
        data_m.recontype = recontype_m;
        data_m.spacing = spacing;
        data_m.wip = wip_m;
        data_m.n_images = sum( prefices == m, 'all') ;
        data{nm} = data_m;
        
    end

end