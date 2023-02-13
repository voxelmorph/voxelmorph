function contour = drawcontour(img, init_contour)
%% drawcontour.m
    if nargin == 1
        init_contour = 0;
    end
    
    while 1
        figure;
        imagesc(img, [0,2e3]); daspect([1, 1, 1]);
        xticks([]); yticks([]);
        if nargin > 1
            hold on;
            plot(init_contour.endo(:, 1), init_contour.endo(:, 2), 'k--');
            plot(init_contour.epi(:, 1), init_contour.epi(:, 2), 'k--');
        end
%         z = input("Use this countour? [yes/no]: ", 's');
%         if strcmpi(z, 'y') || strcmpi(z, 'yes')
%             contour.endo = init_contour.endo;
%             contour.epi = init_contour.epi;
%             close;
%             break;
%         end
        
        p = drawpolygon('LineWidth', 1, 'Color', 'green');
        endo = p.Position;
        p =  drawpolygon('LineWidth', 1, 'Color', 'red');
        epi = p.Position;
        close;
        z = input("Was everything alright? [yes/no]: ", 's');
        if strcmpi(z, 'y') || strcmpi(z, 'yes')
            contour.endo = endo;
            contour.epi = epi;
            break;
        end
    end

end