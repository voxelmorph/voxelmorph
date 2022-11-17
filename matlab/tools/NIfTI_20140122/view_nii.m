%  VIEW_NII: Create or update a 3-View (Front, Top, Side) of the 
%	brain data that is specified by nii structure
%
%  Usage:  	status = view_nii([h], nii, [option])	or
%		status = view_nii(h, [option])
%
%  Where, h is the figure on which the 3-View will be plotted;
%	nii is the brain data in NIFTI format;
%	option is a struct that configures the view plotted, can be:
%
%		option.command = 'init'
%		option.command = 'update'
%		option.command = 'clearnii'
%		option.command = 'updatenii'
%		option.command = 'updateimg' (nii is nii.img here)
%
%		option.usecolorbar = 0 | [1]
%		option.usepanel = 0 | [1]
%		option.usecrosshair = 0 | [1]
%		option.usestretch = 0 | [1]
%		option.useimagesc = 0 | [1]
%		option.useinterp = [0] | 1
%
%		option.setarea = [x y w h] | [0.05 0.05 0.9 0.9]
%		option.setunit = ['vox'] | 'mm'
%		option.setviewpoint = [x y z] | [origin]
%		option.setscanid = [t] | [1]
%		option.setcrosshaircolor = [r g b] | [1 0 0]
%		option.setcolorindex = From 1 to 9 (default is 2 or 3)
%		option.setcolormap = (Mx3 matrix, 0 <= val <= 1)
%		option.setcolorlevel = No more than 256 (default 256)
%		option.sethighcolor = []
%		option.setcbarminmax = []
%		option.setvalue = []
%		option.glblocminmax = []
%		option.setbuttondown = ''
%		option.setcomplex = [0] | 1 | 2
%
%	Options description in detail:
%	==============================
%
%	1. command: A char string that can control program.
%
%		init: If option.command='init', the program will display
%			a 3-View plot on the figure specified by figure h
%			or on a new figure. If there is already a 3-View
%			plot on the figure, please use option.command = 
%			'updatenii' (see detail below); otherwise, the
%			new 3-View plot will superimpose on the old one.
%			If there is no option provided, the program will
%			assume that this is an initial plot. If the figure
%			handle is omitted, the program knows that it is
%			an initial plot.
%
%		update: If there is no command specified, and a figure
%			handle of the existing 3-View plot is provided,
%			the program will choose option.command='update'
%			to update the 3-View plot with some new option 
%			items.
%
%		clearnii: Clear 3-View plot on specific figure
%
%		updatenii: If a new nii is going to be loaded on a fig
%			that has already 3-View plot on it, use this
%			command to clear existing 3-View plot, and then
%			display with new nii. So, the new nii will not
%			superimpose on the existing one. All options
%			for 'init' can be used for 'updatenii'.
%
%		updateimg: If a new 3D matrix with the same dimension
%			is going to be loaded, option.command='updateimg'
%			can be used as a light-weighted 'updatenii, since
%			it only updates the 3 slices with new values.
%			inputing argument nii should be a 3D matrix
%			(nii.img) instead of nii struct. No other option
%			should be used together with 'updateimg' to keep
%			this command as simple as possible.
%
%
%	2. usecolorbar: If specified and usecolorbar=0, the program
%		will not include the colorbar in plot area; otherwise,
%		a colorbar will be included in plot area.
%
%	3. usepanel: If specified and usepanel=0, the control panel
%		at lower right cornor will be invisible; otherwise,
%		it will be visible.
%
%	4. usecrosshair: If specified and usecrosshair=0, the crosshair
%		will be invisible; otherwise, it will be visible.
%
%	5. usestretch: If specified and usestretch=0, the 3 slices will
%		not be stretched, and will be displayed according to
%		the actual voxel size; otherwise, the 3 slices will be 
%		stretched to the edge.
%
%	6. useimagesc: If specified and useimagesc=0, images data will
%		be used directly to match the colormap (like 'image'
%		command); otherwise, image data will be scaled to full
%		colormap with 'imagesc' command in Matlab.
%
%	7. useinterp: If specified and useinterp=1, the image will be
%		displayed using interpolation. Otherwise, it will be
%		displayed like mosaic, and each tile stands for a
%		pixel. This option does not apply to 'setvalue' option
%		is set.
%
%
%	8. setarea: 3-View plot will be displayed on this specific 
%		region. If it is not specified, program will set the
%		plot area to [0.05 0.05 0.9 0.9].
%
%	9. setunit: It can be specified to setunit='voxel' or 'mm'
%		and the view will change the axes unit of [X Y Z]
%		accordingly.
%
%	10. setviewpoint: If specified, [X Y Z] values will be used
%		to set the viewpoint of 3-View plot.
%
%	11. setscanid: If specified, [t] value will be used to display
%		the specified image scan in NIFTI data.
%
%	12. setcrosshaircolor: If specified, [r g b] value will be used
%		for Crosshair Color. Otherwise, red will be the default.
%
%	13. setcolorindex: If specified, the 3-View will choose the
%		following colormap:  2 - Bipolar; 3 - Gray; 4 - Jet;
%		5 - Cool; 6 - Bone; 7 - Hot; 8 - Copper; 9 - Pink;
%		If not specified, it will choose 3 - Gray if all data
%		values are not less than 0; otherwise, it will choose
%		2 - Bipolar if there is value less than 0. (Contrast
%		control can only apply to 3 - Gray colormap.
%
%	14. setcolormap: 3-View plot will use it as a customized colormap.
%		It is a 3-column matrix with value between 0 and 1. If
%		using MS-Windows version of Matlab, the number of rows
%		can not be more than 256, because of Matlab limitation.
%		When colormap is used, setcolorlevel option will be
%		disabled automatically.
%
%	15. setcolorlevel: If specified (must be no more than 256, and
%		cannot be used for customized colormap), row number of
%		colormap will be squeezed down to this level; otherwise,
%		it will assume that setcolorlevel=256.
%
%	16. sethighcolor: If specified, program will squeeze down the
%		colormap, and allocate sethighcolor (an Mx3 matrix)
%		to high-end portion of the colormap. The sum of M and
%		setcolorlevel should be less than 256. If setcolormap
%		option is used, sethighcolor will be inserted on top 
%		of the setcolormap, and the setcolorlevel option will
%		be disabled automatically.
%
%	17. setcbarminmax: if specified, the [min max] will be used to
%		set the min and max of the colorbar, which does not
%		include any data for highcolor.
%
%	18. setvalue: If specified, setvalue.val (with the same size as
%		the source data on solution points) in the source area
%		setvalue.idx will be superimposed on the current nii 
%		image. So, the size of setvalue.val should be equal to
%		the size of setvalue.idx. To use this feature, it needs
%		single or double nii structure for background image.
%
%	19. glblocminmax: If specified, pgm will use glblocminmax to
%		calculate the colormap, instead of minmax of image.
%
%	20. setbuttondown: If specified, pgm will evaluate the command
%		after a click or slide action is invoked to the new
%		view point.
%
%	21. setcomplex: This option will decide how complex data to be
%		displayed:  0 - Real part of complex data; 1 - Imaginary
%		part of complex data; 2 - Modulus (magnitude) of complex
%		data;  If not specified, it will be set to 0 (Real part
%		of complex data as default option. This option only apply
%		when option.command is set to 'init or 'updatenii'.
%
%
%	Additional Options for 'update' command:
%	=======================================
%
%		option.enablecursormove = [1] | 0
%		option.enableviewpoint = 0 | [1]
%		option.enableorigin = 0 | [1]
%		option.enableunit = 0 | [1]
%		option.enablecrosshair = 0 | [1]
%		option.enablehistogram = 0 | [1]
%		option.enablecolormap = 0 | [1]
%		option.enablecontrast = 0 | [1]
%		option.enablebrightness = 0 | [1]
%		option.enableslider = 0 | [1]
%		option.enabledirlabel = 0 | [1]
%
%
%  e.g.:
%	nii = load_nii('T1');		% T1.img/hdr
%	view_nii(nii);
%
%	or
%
%	h = figure('unit','normal','pos', [0.18 0.08 0.64 0.85]);
%	opt.setarea = [0.05 0.05 0.9 0.9];
%	view_nii(h, nii, opt);
%
%
%  Part of this file is copied and modified from:
%  http://www.mathworks.com/matlabcentral/fileexchange/1878-mri-analyze-tools
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function status = view_nii(varargin)

   if nargin < 1
      error('Please check inputs using ''help view_nii''');
   end;

   nii = '';
   opt = '';
   command = '';

   usecolorbar = [];
   usepanel = [];
   usecrosshair = '';
   usestretch = [];
   useimagesc = [];
   useinterp = [];

   setarea = [];
   setunit = '';
   setviewpoint = [];
   setscanid = [];
   setcrosshaircolor = [];
   setcolorindex = '';
   setcolormap = 'NA';
   setcolorlevel = [];
   sethighcolor = 'NA';
   setcbarminmax = [];
   setvalue = [];
   glblocminmax = [];
   setbuttondown = '';
   setcomplex = 0;

   status = [];

   if ishandle(varargin{1})		% plot on top of this figure

      fig = varargin{1};

      if nargin < 2
         command = 'update';		% just to get 3-View status
      end

      if nargin == 2
         if ~isstruct(varargin{2})
            error('2nd parameter should be either nii struct or option struct');
         end

         opt = varargin{2};

         if isfield(opt,'hdr') & isfield(opt,'img')
            nii = opt;
         elseif isfield(opt, 'command') & (strcmpi(opt.command,'init') ...
		| strcmpi(opt.command,'updatenii') ...
		| strcmpi(opt.command,'updateimg') )

            error('Option here cannot contain "init", "updatenii", or "updateimg" comand');
         end
      end

      if nargin == 3
         nii = varargin{2};
         opt = varargin{3};

         if ~isstruct(opt)
            error('3rd parameter should be option struct');
         end

         if ~isfield(opt,'command') | ~strcmpi(opt.command,'updateimg')
            if ~isstruct(nii) | ~isfield(nii,'hdr') | ~isfield(nii,'img')
               error('2nd parameter should be nii struct');
            end

            if isfield(nii,'untouch') & nii.untouch == 1
               error('Usage: please use ''load_nii.m'' to load the structure.');
            end
         end
      end

      set(fig, 'menubar', 'none');

   elseif ischar(varargin{1})		% call back by event

      command = lower(varargin{1});
      fig = gcbf;

   else					% start nii with a new figure

      nii = varargin{1};

      if ~isstruct(nii) | ~isfield(nii,'hdr') | ~isfield(nii,'img')
         error('1st parameter should be either a figure handle or nii struct');
      end

      if isfield(nii,'untouch') & nii.untouch == 1
         error('Usage: please use ''load_nii.m'' to load the structure.');
      end

      if nargin > 1
         opt = varargin{2};

         if isfield(opt, 'command') & ~strcmpi(opt.command,'init')
            error('Option here must use "init" comand');
         end
      end

      command = 'init';
      fig = figure('unit','normal','position',[0.15 0.08 0.70 0.85]);
      view_nii_menu(fig);
      rri_file_menu(fig);

   end

   if ~isempty(opt)

      if isfield(opt,'command')
         command = lower(opt.command);
      end

      if isempty(command)
         command = 'update';
      end

      if isfield(opt,'usecolorbar')
         usecolorbar = opt.usecolorbar;
      end

      if isfield(opt,'usepanel')
         usepanel = opt.usepanel;
      end

      if isfield(opt,'usecrosshair')
         usecrosshair = opt.usecrosshair;
      end

      if isfield(opt,'usestretch')
         usestretch = opt.usestretch;
      end

      if isfield(opt,'useimagesc')
         useimagesc = opt.useimagesc;
      end

      if isfield(opt,'useinterp')
         useinterp = opt.useinterp;
      end

      if isfield(opt,'setarea')
         setarea = opt.setarea;
      end

      if isfield(opt,'setunit')
         setunit = opt.setunit;
      end

      if isfield(opt,'setviewpoint')
         setviewpoint = opt.setviewpoint;
      end

      if isfield(opt,'setscanid')
         setscanid = opt.setscanid;
      end

      if isfield(opt,'setcrosshaircolor')
         setcrosshaircolor = opt.setcrosshaircolor;

         if ~isempty(setcrosshaircolor) & (~isnumeric(setcrosshaircolor) | ~isequal(size(setcrosshaircolor),[1 3]) | min(setcrosshaircolor(:))<0 | max(setcrosshaircolor(:))>1)
            error('Crosshair Color should be a 1x3 matrix with value between 0 and 1');
         end
      end

      if isfield(opt,'setcolorindex')
         setcolorindex = round(opt.setcolorindex);

         if ~isnumeric(setcolorindex) | setcolorindex < 1 | setcolorindex > 9
            error('Colorindex should be a number between 1 and 9');
         end
      end

      if isfield(opt,'setcolormap')
         setcolormap = opt.setcolormap;

         if ~isempty(setcolormap) & (~isnumeric(setcolormap) | size(setcolormap,2) ~= 3 | min(setcolormap(:))<0 | max(setcolormap(:))>1)
            error('Colormap should be a Mx3 matrix with value between 0 and 1');
         end
      end

      if isfield(opt,'setcolorlevel')
         setcolorlevel = round(opt.setcolorlevel);

         if ~isnumeric(setcolorlevel) | setcolorlevel > 256 | setcolorlevel < 1
            error('Colorlevel should be a number between 1 and 256');
         end
      end

      if isfield(opt,'sethighcolor')
         sethighcolor = opt.sethighcolor;

         if ~isempty(sethighcolor) & (~isnumeric(sethighcolor) | size(sethighcolor,2) ~= 3 | min(sethighcolor(:))<0 | max(sethighcolor(:))>1)
            error('Highcolor should be a Mx3 matrix with value between 0 and 1');
         end
      end

      if isfield(opt,'setcbarminmax')
         setcbarminmax = opt.setcbarminmax;

         if isempty(setcbarminmax) | ~isnumeric(setcbarminmax) | length(setcbarminmax) ~= 2
            error('Colorbar MinMax should contain 2 values: [min max]');
         end
      end

      if isfield(opt,'setvalue')
         setvalue = opt.setvalue;

         if isempty(setvalue) | ~isstruct(setvalue) | ...
		~isfield(opt.setvalue,'idx') | ~isfield(opt.setvalue,'val')
            error('setvalue should be a struct contains idx and val');
         end

         if length(opt.setvalue.idx(:)) ~= length(opt.setvalue.val(:))
            error('length of idx and val fields should be the same');
         end

         if ~strcmpi(class(opt.setvalue.idx),'single')
            opt.setvalue.idx = single(opt.setvalue.idx);
         end

         if ~strcmpi(class(opt.setvalue.val),'single')
            opt.setvalue.val = single(opt.setvalue.val);
         end
      end

      if isfield(opt,'glblocminmax')
         glblocminmax = opt.glblocminmax;
      end

      if isfield(opt,'setbuttondown')
         setbuttondown = opt.setbuttondown;
      end

      if isfield(opt,'setcomplex')
         setcomplex = opt.setcomplex;
      end

   end

   switch command

   case {'init'}

      set(fig, 'InvertHardcopy','off');
      set(fig, 'PaperPositionMode','auto');

      fig = init(nii, fig, setarea, setunit, setviewpoint, setscanid, setbuttondown, ...
         setcolorindex, setcolormap, setcolorlevel, sethighcolor, setcbarminmax, ...
         usecolorbar, usepanel, usecrosshair, usestretch, useimagesc, useinterp, ...
         setvalue, glblocminmax, setcrosshaircolor, setcomplex);

      %  get status
      %
      status = get_status(fig);

   case {'update'}

      nii_view = getappdata(fig,'nii_view');
      h = fig;

      if isempty(nii_view)
         error('The figure should already contain a 3-View plot.');
      end

      if ~isempty(opt)

         %  Order of the following update matters.
         %
         update_shape(h, setarea, usecolorbar, usestretch, useimagesc);
         update_useinterp(h, useinterp);
         update_useimagesc(h, useimagesc);
         update_usepanel(h, usepanel);
         update_colorindex(h, setcolorindex);
         update_colormap(h, setcolormap);
         update_highcolor(h, sethighcolor, setcolorlevel);
         update_cbarminmax(h, setcbarminmax);
         update_unit(h, setunit);
         update_viewpoint(h, setviewpoint);
         update_scanid(h, setscanid);
         update_buttondown(h, setbuttondown);
         update_crosshaircolor(h, setcrosshaircolor);
         update_usecrosshair(h, usecrosshair);

         %  Enable/Disable object
         %
         update_enable(h, opt);

      end

      %  get status
      %
      status = get_status(h);

   case {'updateimg'}

      if ~exist('nii','var')
         msg = sprintf('Please input a 3D matrix brain data');
         error(msg);
      end

      %  Note: nii is not nii, nii should be a 3D matrix here
      %
      if ~isnumeric(nii)
         msg = sprintf('2nd parameter should be a 3D matrix, not nii struct');
         error(msg);
      end

      nii_view = getappdata(fig,'nii_view');

      if isempty(nii_view)
         error('The figure should already contain a 3-View plot.');
      end

      img = nii;
      update_img(img, fig, opt);

      %  get status
      %
      status = get_status(fig);

   case {'updatenii'}

      nii_view = getappdata(fig,'nii_view');

      if isempty(nii_view)
         error('The figure should already contain a 3-View plot.');
      end

      if ~isstruct(nii) | ~isfield(nii,'hdr') | ~isfield(nii,'img')
         error('2nd parameter should be nii struct');
      end

      if isfield(nii,'untouch') & nii.untouch == 1
         error('Usage: please use ''load_nii.m'' to load the structure.');
      end

      opt.command = 'clearnii';
      view_nii(fig, opt);

      opt.command = 'init';
      view_nii(fig, nii, opt);

      %  get status
      %
      status = get_status(fig);

   case {'clearnii'}

      nii_view = getappdata(fig,'nii_view');

      handles = struct2cell(nii_view.handles);

      for i=1:length(handles)
         if ishandle(handles{i})	% in case already del by parent
            delete(handles{i});
         end
      end

      rmappdata(fig,'nii_view');
      buttonmotion = get(fig,'windowbuttonmotion');
      mymotion = '; view_nii(''move_cursor'');';
      buttonmotion = strrep(buttonmotion, mymotion, '');
      set(fig, 'windowbuttonmotion', buttonmotion);

   case {'axial_image','coronal_image','sagittal_image'}    

      switch command
         case 'axial_image',    view = 'axi'; axi = 0; cor = 1; sag = 1;
         case 'coronal_image',  view = 'cor'; axi = 1; cor = 0; sag = 1;
         case 'sagittal_image', view = 'sag'; axi = 1; cor = 1; sag = 0;
      end

      nii_view = getappdata(fig,'nii_view');
      nii_view = get_slice_position(nii_view,view);

      if isfield(nii_view, 'disp')
         img = nii_view.disp;
      else
         img = nii_view.nii.img;
      end

      %  CData must be double() for Matlab 6.5 for Windows
      %
      if axi,
         if isfield(nii_view.handles,'axial_bg') & ~isempty(nii_view.handles.axial_bg) & nii_view.useinterp
            Saxi = squeeze(nii_view.bgimg(:,:,nii_view.slices.axi));
            set(nii_view.handles.axial_bg,'CData',double(Saxi)');
         end

         if isfield(nii_view.handles,'axial_image'),
            if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
               Saxi = squeeze(img(:,:,nii_view.slices.axi,:,nii_view.scanid));
               Saxi = permute(Saxi, [2 1 3]);
            else
               Saxi = squeeze(img(:,:,nii_view.slices.axi,nii_view.scanid));
               Saxi = Saxi';
            end

            set(nii_view.handles.axial_image,'CData',double(Saxi));
         end

         if isfield(nii_view.handles,'axial_slider'),
            set(nii_view.handles.axial_slider,'Value',nii_view.slices.axi);
         end;
       end

       if cor,
         if isfield(nii_view.handles,'coronal_bg') & ~isempty(nii_view.handles.coronal_bg) & nii_view.useinterp
            Scor = squeeze(nii_view.bgimg(:,nii_view.slices.cor,:));
            set(nii_view.handles.coronal_bg,'CData',double(Scor)');
         end

         if isfield(nii_view.handles,'coronal_image'),
            if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
               Scor = squeeze(img(:,nii_view.slices.cor,:,:,nii_view.scanid));
               Scor = permute(Scor, [2 1 3]);
            else
               Scor = squeeze(img(:,nii_view.slices.cor,:,nii_view.scanid));
               Scor = Scor';
            end

            set(nii_view.handles.coronal_image,'CData',double(Scor));
         end

         if isfield(nii_view.handles,'coronal_slider'),
            slider_val = nii_view.dims(2) - nii_view.slices.cor + 1;
            set(nii_view.handles.coronal_slider,'Value',slider_val);
         end;
      end;

      if sag,
         if isfield(nii_view.handles,'sagittal_bg') & ~isempty(nii_view.handles.sagittal_bg) & nii_view.useinterp
            Ssag = squeeze(nii_view.bgimg(nii_view.slices.sag,:,:));
            set(nii_view.handles.sagittal_bg,'CData',double(Ssag)');
         end

         if isfield(nii_view.handles,'sagittal_image'),
            if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
               Ssag = squeeze(img(nii_view.slices.sag,:,:,:,nii_view.scanid));
               Ssag = permute(Ssag, [2 1 3]);
            else
               Ssag = squeeze(img(nii_view.slices.sag,:,:,nii_view.scanid));
               Ssag = Ssag';
            end

            set(nii_view.handles.sagittal_image,'CData',double(Ssag));
         end

         if isfield(nii_view.handles,'sagittal_slider'),
            set(nii_view.handles.sagittal_slider,'Value',nii_view.slices.sag);
         end;
      end;

      update_nii_view(nii_view);

      if ~isempty(nii_view.buttondown)
         eval(nii_view.buttondown);
      end
    
   case {'axial_slider','coronal_slider','sagittal_slider'},
    
      switch command
         case 'axial_slider',    view = 'axi'; axi = 1; cor = 0; sag = 0;
         case 'coronal_slider',  view = 'cor'; axi = 0; cor = 1; sag = 0;
         case 'sagittal_slider', view = 'sag'; axi = 0; cor = 0; sag = 1;
      end

      nii_view = getappdata(fig,'nii_view');
      nii_view = get_slider_position(nii_view);

      if isfield(nii_view, 'disp')
         img = nii_view.disp;
      else
         img = nii_view.nii.img;
      end

      if axi,
         if isfield(nii_view.handles,'axial_bg') & ~isempty(nii_view.handles.axial_bg) & nii_view.useinterp
            Saxi = squeeze(nii_view.bgimg(:,:,nii_view.slices.axi));
            set(nii_view.handles.axial_bg,'CData',double(Saxi)');
         end

         if isfield(nii_view.handles,'axial_image'),
            if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
               Saxi = squeeze(img(:,:,nii_view.slices.axi,:,nii_view.scanid));
               Saxi = permute(Saxi, [2 1 3]);
            else
               Saxi = squeeze(img(:,:,nii_view.slices.axi,nii_view.scanid));
               Saxi = Saxi';
            end

            set(nii_view.handles.axial_image,'CData',double(Saxi));
         end

         if isfield(nii_view.handles,'axial_slider'),
            set(nii_view.handles.axial_slider,'Value',nii_view.slices.axi);
         end
      end

      if cor,
         if isfield(nii_view.handles,'coronal_bg') & ~isempty(nii_view.handles.coronal_bg) & nii_view.useinterp
            Scor = squeeze(nii_view.bgimg(:,nii_view.slices.cor,:));
            set(nii_view.handles.coronal_bg,'CData',double(Scor)');
         end

         if isfield(nii_view.handles,'coronal_image'),
            if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
               Scor = squeeze(img(:,nii_view.slices.cor,:,:,nii_view.scanid));
               Scor = permute(Scor, [2 1 3]);
            else
               Scor = squeeze(img(:,nii_view.slices.cor,:,nii_view.scanid));
               Scor = Scor';
            end

            set(nii_view.handles.coronal_image,'CData',double(Scor));
         end

         if isfield(nii_view.handles,'coronal_slider'),
            slider_val = nii_view.dims(2) - nii_view.slices.cor + 1;
            set(nii_view.handles.coronal_slider,'Value',slider_val);
         end
      end    

      if sag,
         if isfield(nii_view.handles,'sagittal_bg') & ~isempty(nii_view.handles.sagittal_bg) & nii_view.useinterp
            Ssag = squeeze(nii_view.bgimg(nii_view.slices.sag,:,:));
            set(nii_view.handles.sagittal_bg,'CData',double(Ssag)');
         end

         if isfield(nii_view.handles,'sagittal_image'),
            if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
               Ssag = squeeze(img(nii_view.slices.sag,:,:,:,nii_view.scanid));
               Ssag = permute(Ssag, [2 1 3]);
            else
               Ssag = squeeze(img(nii_view.slices.sag,:,:,nii_view.scanid));
               Ssag = Ssag';
            end

            set(nii_view.handles.sagittal_image,'CData',double(Ssag));
         end

         if isfield(nii_view.handles,'sagittal_slider'),
            set(nii_view.handles.sagittal_slider,'Value',nii_view.slices.sag);
         end
      end

      update_nii_view(nii_view);

      if ~isempty(nii_view.buttondown)
         eval(nii_view.buttondown);
      end

   case {'impos_edit'}

      nii_view = getappdata(fig,'nii_view');
      impos = str2num(get(nii_view.handles.impos,'string'));

      if isfield(nii_view, 'disp')
         img = nii_view.disp;
      else
         img = nii_view.nii.img;
      end

      if isempty(impos) | ~all(size(impos) ==  [1 3])
         msg = 'Please use 3 numbers to represent X,Y and Z';
         msgbox(msg,'Error');
         return;
      end

      slices.sag = round(impos(1));
      slices.cor = round(impos(2));
      slices.axi = round(impos(3));

      nii_view = convert2voxel(nii_view,slices);
      nii_view = check_slices(nii_view);

      impos(1) = nii_view.slices.sag;
      impos(2) = nii_view.dims(2) - nii_view.slices.cor + 1;
      impos(3) = nii_view.slices.axi;

      if isfield(nii_view.handles,'sagittal_slider'),
         set(nii_view.handles.sagittal_slider,'Value',impos(1));
      end

      if isfield(nii_view.handles,'coronal_slider'),
         set(nii_view.handles.coronal_slider,'Value',impos(2));
      end

      if isfield(nii_view.handles,'axial_slider'),
         set(nii_view.handles.axial_slider,'Value',impos(3));
      end

      nii_view = get_slider_position(nii_view);
      update_nii_view(nii_view);

      if isfield(nii_view.handles,'axial_bg') & ~isempty(nii_view.handles.axial_bg) & nii_view.useinterp
         Saxi = squeeze(nii_view.bgimg(:,:,nii_view.slices.axi));
         set(nii_view.handles.axial_bg,'CData',double(Saxi)');
      end

      if isfield(nii_view.handles,'axial_image'),
         if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
            Saxi = squeeze(img(:,:,nii_view.slices.axi,:,nii_view.scanid));
            Saxi = permute(Saxi, [2 1 3]);
         else
            Saxi = squeeze(img(:,:,nii_view.slices.axi,nii_view.scanid));
            Saxi = Saxi';
         end

         set(nii_view.handles.axial_image,'CData',double(Saxi));
      end

      if isfield(nii_view.handles,'axial_slider'),
         set(nii_view.handles.axial_slider,'Value',nii_view.slices.axi);
      end

      if isfield(nii_view.handles,'coronal_bg') & ~isempty(nii_view.handles.coronal_bg) & nii_view.useinterp
         Scor = squeeze(nii_view.bgimg(:,nii_view.slices.cor,:));
         set(nii_view.handles.coronal_bg,'CData',double(Scor)');
      end

      if isfield(nii_view.handles,'coronal_image'),
         if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
            Scor = squeeze(img(:,nii_view.slices.cor,:,:,nii_view.scanid));
            Scor = permute(Scor, [2 1 3]);
         else
            Scor = squeeze(img(:,nii_view.slices.cor,:,nii_view.scanid));
            Scor = Scor';
         end

         set(nii_view.handles.coronal_image,'CData',double(Scor));
      end

      if isfield(nii_view.handles,'coronal_slider'),
         slider_val = nii_view.dims(2) - nii_view.slices.cor + 1;
         set(nii_view.handles.coronal_slider,'Value',slider_val);
      end

      if isfield(nii_view.handles,'sagittal_bg') & ~isempty(nii_view.handles.sagittal_bg) & nii_view.useinterp
         Ssag = squeeze(nii_view.bgimg(nii_view.slices.sag,:,:));
         set(nii_view.handles.sagittal_bg,'CData',double(Ssag)');
      end

      if isfield(nii_view.handles,'sagittal_image'),
         if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
            Ssag = squeeze(img(nii_view.slices.sag,:,:,:,nii_view.scanid));
            Ssag = permute(Ssag, [2 1 3]);
         else
            Ssag = squeeze(img(nii_view.slices.sag,:,:,nii_view.scanid));
            Ssag = Ssag';
         end

         set(nii_view.handles.sagittal_image,'CData',double(Ssag));
      end

      if isfield(nii_view.handles,'sagittal_slider'),
         set(nii_view.handles.sagittal_slider,'Value',nii_view.slices.sag);
      end

      axes(nii_view.handles.axial_axes);
      axes(nii_view.handles.coronal_axes);
      axes(nii_view.handles.sagittal_axes);

      if ~isempty(nii_view.buttondown)
         eval(nii_view.buttondown);
      end

   case 'coordinates',

      nii_view = getappdata(fig,'nii_view');
      set_image_value(nii_view);

   case 'crosshair',

      nii_view = getappdata(fig,'nii_view');

      if get(nii_view.handles.xhair,'value') == 2		% off
         set(nii_view.axi_xhair.lx,'visible','off');
         set(nii_view.axi_xhair.ly,'visible','off');
         set(nii_view.cor_xhair.lx,'visible','off');
         set(nii_view.cor_xhair.ly,'visible','off');
         set(nii_view.sag_xhair.lx,'visible','off');
         set(nii_view.sag_xhair.ly,'visible','off');
      else
         set(nii_view.axi_xhair.lx,'visible','on');
         set(nii_view.axi_xhair.ly,'visible','on');
         set(nii_view.cor_xhair.lx,'visible','on');
         set(nii_view.cor_xhair.ly,'visible','on');
         set(nii_view.sag_xhair.lx,'visible','on');
         set(nii_view.sag_xhair.ly,'visible','on');

         set(nii_view.handles.axial_axes,'selected','on');
         set(nii_view.handles.axial_axes,'selected','off');
         set(nii_view.handles.coronal_axes,'selected','on');
         set(nii_view.handles.coronal_axes,'selected','off');
         set(nii_view.handles.sagittal_axes,'selected','on');
         set(nii_view.handles.sagittal_axes,'selected','off');
      end

   case 'xhair_color',

      old_color = get(gcbo,'user');
      new_color = uisetcolor(old_color);
      update_crosshaircolor(fig, new_color);

   case {'color','contrast_def'}

      nii_view = getappdata(fig,'nii_view');

      if nii_view.numscan == 1
         if get(nii_view.handles.colorindex,'value') == 2
            set(nii_view.handles.contrast,'value',128);
         elseif get(nii_view.handles.colorindex,'value') == 3
            set(nii_view.handles.contrast,'value',1);
         end
      end

      [custom_color_map, custom_colorindex] = change_colormap(fig);

      if strcmpi(command, 'color')

         setcolorlevel = nii_view.colorlevel;

         if ~isempty(custom_color_map)		% isfield(nii_view, 'color_map')
            setcolormap = custom_color_map;	% nii_view.color_map;
         else
            setcolormap = [];
         end

         if isfield(nii_view, 'highcolor')
            sethighcolor = nii_view.highcolor;
         else
            sethighcolor = [];
         end

         redraw_cbar(fig, setcolorlevel, setcolormap, sethighcolor);

         if nii_view.numscan == 1 & ...
		(custom_colorindex < 2 | custom_colorindex > 3)
            contrastopt.enablecontrast = 0;
         else
            contrastopt.enablecontrast = 1;
         end

         update_enable(fig, contrastopt);

      end

   case {'neg_color','brightness','contrast'}

      change_colormap(fig);

   case {'brightness_def'}

      nii_view = getappdata(fig,'nii_view');
      set(nii_view.handles.brightness,'value',0);
      change_colormap(fig);

   case 'hist_plot'

      hist_plot(fig);

   case 'hist_eq'

      hist_eq(fig);

   case 'move_cursor'

      move_cursor(fig);

   case 'edit_change_scan'

      change_scan('edit_change_scan');

   case 'slider_change_scan'

      change_scan('slider_change_scan');

   end

   return;						% view_nii


%----------------------------------------------------------------
function fig = init(nii, fig, area, setunit, setviewpoint, setscanid, buttondown, ...
         colorindex, color_map, colorlevel, highcolor, cbarminmax, ...
         usecolorbar, usepanel, usecrosshair, usestretch, useimagesc, ...
         useinterp, setvalue, glblocminmax, setcrosshaircolor, ...
         setcomplex)

   %  Support data type COMPLEX64 & COMPLEX128
   %
   if nii.hdr.dime.datatype == 32 | nii.hdr.dime.datatype == 1792
      switch setcomplex,
      case 0,
         nii.img = real(nii.img);
      case 1,
         nii.img = imag(nii.img);
      case 2,
         if isa(nii.img, 'double')
            nii.img = abs(double(nii.img));
         else
            nii.img = single(abs(double(nii.img)));
         end
      end
   end

   if isempty(area)
      area = [0.05 0.05 0.9 0.9];
   end

   if isempty(setscanid)
      setscanid = 1;
   else
      setscanid = round(setscanid);

      if setscanid < 1
         setscanid = 1;
      end

      if setscanid > nii.hdr.dime.dim(5)
         setscanid = nii.hdr.dime.dim(5);
      end
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      usecolorbar = 0;
   elseif isempty(usecolorbar)
      usecolorbar = 1;
   end

   if isempty(usepanel)
      usepanel = 1;
   end

   if isempty(usestretch)
      usestretch = 1;
   end

   if isempty(useimagesc)
      useimagesc = 1;
   end

   if isempty(useinterp)
      useinterp = 0;
   end

   if isempty(colorindex)
      tmp = min(nii.img(:,:,:,setscanid));

      if  min(tmp(:)) < 0
         colorindex = 2;
         setcrosshaircolor = [1 1 0];
      else
         colorindex = 3;
      end
   end

   if isempty(color_map) | ischar(color_map)
      color_map = [];
   else
      colorindex = 1;
   end

   bgimg = [];

   if ~isempty(glblocminmax)
      minvalue = glblocminmax(1);
      maxvalue = glblocminmax(2);
   else
      minvalue = nii.img(:,:,:,setscanid);
      minvalue = double(minvalue(:));
      minvalue = min(minvalue(~isnan(minvalue)));
      maxvalue = nii.img(:,:,:,setscanid);
      maxvalue = double(maxvalue(:));
      maxvalue = max(maxvalue(~isnan(maxvalue)));
   end

   if ~isempty(setvalue)
      if ~isempty(glblocminmax)
         minvalue = glblocminmax(1);
         maxvalue = glblocminmax(2);
      else
         minvalue = double(min(setvalue.val));
         maxvalue = double(max(setvalue.val));
      end

      bgimg = double(nii.img);
      minbg = double(min(bgimg(:)));
      maxbg = double(max(bgimg(:)));

      bgimg = scale_in(bgimg, minbg, maxbg, 55) + 200;	% scale to 201~256

      %  56 level for brain structure
      %
%      highcolor = [zeros(1,3);gray(55)];
      highcolor = gray(56);
      cbarminmax = [minvalue maxvalue];

      if useinterp

         %  scale signal data to 1~200
         %
         nii.img = repmat(nan, size(nii.img));
         nii.img(setvalue.idx) = setvalue.val;

         %  200 level for source image
         %
         bgimg = single(scale_out(bgimg, cbarminmax(1), cbarminmax(2), 199));
      else

         bgimg(setvalue.idx) = NaN;
         minbg = double(min(bgimg(:)));
         maxbg = double(max(bgimg(:)));
         bgimg(setvalue.idx) = minbg;

         %  bgimg must be normalized to [201 256]
         %
         bgimg = 55 * (bgimg-min(bgimg(:))) / (max(bgimg(:))-min(bgimg(:))) + 201;
         bgimg(setvalue.idx) = 0;

         %  scale signal data to 1~200
         %
         nii.img = zeros(size(nii.img));
         nii.img(setvalue.idx) = scale_in(setvalue.val, minvalue, maxvalue, 199);
         nii.img = nii.img + bgimg;
         bgimg = [];
         nii.img = scale_out(nii.img, cbarminmax(1), cbarminmax(2), 199);

         minvalue = double(nii.img(:));
         minvalue = min(minvalue(~isnan(minvalue)));
         maxvalue = double(nii.img(:));
         maxvalue = max(maxvalue(~isnan(maxvalue)));

         if ~isempty(glblocminmax)		% maxvalue is gray
            minvalue = glblocminmax(1);
         end

      end

      colorindex = 2;
      setcrosshaircolor = [1 1 0];

   end

   if isempty(highcolor) | ischar(highcolor)
      highcolor = [];
      num_highcolor = 0;
   else
      num_highcolor = size(highcolor,1);
   end

   if isempty(colorlevel)
      colorlevel = 256 - num_highcolor;
   end

   if usecolorbar
      cbar_area = area;
      cbar_area(1) = area(1) + area(3)*0.93;
      cbar_area(3) = area(3)*0.04;
      area(3) = area(3)*0.9;		% 90% used for main axes
   else
      cbar_area = [];
   end

   %  init color (gray) scaling to make sure the slice clim take the
   %  global clim [min(nii.img(:)) max(nii.img(:))]
   %
   if isempty(bgimg)
      clim = [minvalue maxvalue];
   else
      clim = [minvalue double(max(bgimg(:)))];
   end

   if clim(1) == clim(2)
      clim(2) = clim(1) + 0.000001;
   end

   if isempty(cbarminmax)
      cbarminmax = [minvalue maxvalue];
   end

   xdim = size(nii.img, 1);
   ydim = size(nii.img, 2);
   zdim = size(nii.img, 3);

   dims = [xdim ydim zdim];
   voxel_size = abs(nii.hdr.dime.pixdim(2:4));		% vol in mm

   if any(voxel_size <= 0)
      voxel_size(find(voxel_size <= 0)) = 1;
   end

   origin = abs(nii.hdr.hist.originator(1:3));

   if isempty(origin) | all(origin == 0)		% according to SPM
      origin = (dims+1)/2;   
   end;

   origin = round(origin);

   if any(origin > dims)				% simulate fMRI
      origin(find(origin > dims)) = dims(find(origin > dims));
   end

   if any(origin <= 0)
      origin(find(origin <= 0)) = 1;
   end

   nii_view.dims = dims;
   nii_view.voxel_size = voxel_size;
   nii_view.origin = origin;

   nii_view.slices.sag = 1;
   nii_view.slices.cor = 1;
   nii_view.slices.axi = 1;
   if xdim > 1, nii_view.slices.sag = origin(1); end
   if ydim > 1, nii_view.slices.cor = origin(2); end
   if zdim > 1, nii_view.slices.axi = origin(3); end

   nii_view.area = area;
   nii_view.fig = fig;
   nii_view.nii = nii;					% image data
   nii_view.bgimg = bgimg;				% background
   nii_view.setvalue = setvalue;
   nii_view.minvalue = minvalue;
   nii_view.maxvalue = maxvalue;
   nii_view.numscan = nii.hdr.dime.dim(5);
   nii_view.scanid = setscanid;

   Font.FontUnits  = 'point';
   Font.FontSize   = 12;

   %  create axes for colorbar
   %
   [cbar_axes cbarminmax_axes] = create_cbar_axes(fig, cbar_area);

   if isempty(cbar_area)
      nii_view.cbar_area = [];
   else
      nii_view.cbar_area = cbar_area;
   end

   %  create axes for top/front/side view
   %
   vol_size = voxel_size .* dims;
   [top_ax, front_ax, side_ax] ...
	= create_ax(fig, area, vol_size, usestretch);

   top_pos = get(top_ax,'position');
   front_pos = get(front_ax,'position');
   side_pos = get(side_ax,'position');

   %  Sagittal Slider
   %
   x = side_pos(1);
   y = top_pos(2) + top_pos(4);
   w = side_pos(3);
   h = (front_pos(2) - y) / 2;
   y = y + h;

   pos = [x y w h];

   if xdim > 1,
      slider_step(1) = 1/(xdim);
      slider_step(2) = 1.00001/(xdim);

      handles.sagittal_slider = uicontrol('Parent',fig, ...
                'Style','slider','Units','Normalized', Font, ...
                'Position',pos, 'HorizontalAlignment','center',...
                'BackgroundColor',[0.5 0.5 0.5],'ForegroundColor',[0 0 0],...
                'BusyAction','queue',...
                'TooltipString','Sagittal slice navigation',...
                'Min',1,'Max',xdim,'SliderStep',slider_step, ...
                'Value',nii_view.slices.sag,...
                'Callback','view_nii(''sagittal_slider'');');

      set(handles.sagittal_slider,'position',pos);	% linux66
   end

   %  Coronal Slider
   %
   x = top_pos(1);
   y = top_pos(2) + top_pos(4);
   w = top_pos(3);
   h = (front_pos(2) - y) / 2;
   y = y + h;

   pos = [x y w h];

   if ydim > 1,
      slider_step(1) = 1/(ydim);
      slider_step(2) = 1.00001/(ydim);

      slider_val = nii_view.dims(2) - nii_view.slices.cor + 1;

      handles.coronal_slider = uicontrol('Parent',fig, ...
                'Style','slider','Units','Normalized', Font, ...
                'Position',pos, 'HorizontalAlignment','center',...
                'BackgroundColor',[0.5 0.5 0.5],'ForegroundColor',[0 0 0],...
                'BusyAction','queue',...
                'TooltipString','Coronal slice navigation',...
                'Min',1,'Max',ydim,'SliderStep',slider_step, ...
                'Value',slider_val,...
                'Callback','view_nii(''coronal_slider'');');

      set(handles.coronal_slider,'position',pos);	% linux66
   end

   %  Axial Slider
   %
%   x = front_pos(1) + front_pos(3);
%   y = front_pos(2);
%   w = side_pos(1) - x;
%   h = front_pos(4);

   x = top_pos(1);
   y = area(2);
   w = top_pos(3);
   h = top_pos(2) - y;

   pos = [x y w h];

   if zdim > 1,
      slider_step(1) = 1/(zdim);
      slider_step(2) = 1.00001/(zdim);

      handles.axial_slider = uicontrol('Parent',fig, ...
                'Style','slider','Units','Normalized', Font, ...
                'Position',pos, 'HorizontalAlignment','center',...
                'BackgroundColor',[0.5 0.5 0.5],'ForegroundColor',[0 0 0],...
                'BusyAction','queue',...
                'TooltipString','Axial slice navigation',...
                'Min',1,'Max',zdim,'SliderStep',slider_step, ...
                'Value',nii_view.slices.axi,...
                'Callback','view_nii(''axial_slider'');');

      set(handles.axial_slider,'position',pos);	% linux66
   end

   %  plot info view
   %
%   info_pos = [side_pos([1,3]); top_pos([2,4])];
%   info_pos = info_pos(:);
   gap = side_pos(1)-(top_pos(1)+top_pos(3));
   info_pos(1) = side_pos(1) + gap;
   info_pos(2) = area(2);
   info_pos(3) = side_pos(3) - gap;
   info_pos(4) = top_pos(2) + top_pos(4) - area(2) - gap;

   num_inputline = 10;
   inputline_space =info_pos(4) / num_inputline;


   %  for any info_area change, update_usestretch should also be changed


   %  Image Intensity Value at Cursor
   %
   x = info_pos(1);
   y = info_pos(2);
   w = info_pos(3)*0.5;
   h = inputline_space*0.6;

   pos = [x y w h];

   handles.Timvalcur = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'left',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Value at cursor:');

   if usepanel
      set(handles.Timvalcur, 'visible', 'on');
   end

   x = x + w;
   w = info_pos(3)*0.5;

   pos = [x y w h];

   handles.imvalcur = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'right',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String',' ');

   if usepanel
      set(handles.imvalcur, 'visible', 'on');
   end

   %  Position at Cursor
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.5;

   pos = [x y w h];

   handles.Timposcur = uicontrol('Parent',fig,'Style','text', ...
        'Units','Normalized', Font, ...
        'Position',pos, 'HorizontalAlignment', 'left',...
        'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
        'BusyAction','queue',...
        'visible','off', ...
        'String','[X Y Z] at cursor:');

   if usepanel
      set(handles.Timposcur, 'visible', 'on');
   end

   x = x + w;
   w = info_pos(3)*0.5;

   pos = [x y w h];

   handles.imposcur = uicontrol('Parent',fig,'Style','text', ...
        'Units','Normalized', Font, ...
        'Position',pos, 'HorizontalAlignment', 'right',...
        'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
        'BusyAction','queue',...
        'visible','off', ...
        'String',' ','Value',[0 0 0]);

   if usepanel
      set(handles.imposcur, 'visible', 'on');
   end

   %  Image Intensity Value at Mouse Click
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.5;

   pos = [x y w h];

   handles.Timval = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'left',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Value at crosshair:');

   if usepanel
      set(handles.Timval, 'visible', 'on');
   end

   x = x + w;
   w = info_pos(3)*0.5;

   pos = [x y w h];

   handles.imval = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'right',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String',' ');

   if usepanel
      set(handles.imval, 'visible', 'on');
   end

   %  Viewpoint Position at Mouse Click
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.5;

   pos = [x y w h];

   handles.Timpos = uicontrol('Parent',fig,'Style','text', ...
        'Units','Normalized', Font, ...
        'Position',pos, 'HorizontalAlignment', 'left',...
        'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
        'BusyAction','queue',...
        'visible','off', ...
        'String','[X Y Z] at crosshair:');

   if usepanel
      set(handles.Timpos, 'visible', 'on');
   end

   x = x + w + 0.005;
   y = y - 0.008;
   w = info_pos(3)*0.5;
   h = inputline_space*0.9;

   pos = [x y w h];

   handles.impos = uicontrol('Parent',fig,'Style','edit', ...
        'Units','Normalized', Font, ...
        'Position',pos, 'HorizontalAlignment', 'right',...
        'BackgroundColor', [1 1 1], 'ForegroundColor', [0 0 0],...
        'BusyAction','queue',...
        'Callback','view_nii(''impos_edit'');', ...
        'TooltipString','Viewpoint Location in Axes Unit', ...
        'visible','off', ...
        'String',' ','Value',[0 0 0]);

   if usepanel
      set(handles.impos, 'visible', 'on');
   end

   %  Origin Position
   %
   x = info_pos(1);
   y = y + inputline_space*1.2;
   w = info_pos(3)*0.5;
   h = inputline_space*0.6;

   pos = [x y w h];

   handles.Torigin = uicontrol('Parent',fig,'Style','text', ...
        'Units','Normalized', Font, ...
        'Position',pos, 'HorizontalAlignment', 'left',...
        'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
        'BusyAction','queue',...
        'visible','off', ...
        'String','[X Y Z] at origin:');

   if usepanel
      set(handles.Torigin, 'visible', 'on');
   end

   x = x + w;
   w = info_pos(3)*0.5;

   pos = [x y w h];

   handles.origin = uicontrol('Parent',fig,'Style','text', ...
        'Units','Normalized', Font, ...
        'Position',pos, 'HorizontalAlignment', 'right',...
        'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
        'BusyAction','queue',...
        'visible','off', ...
        'String',' ','Value',[0 0 0]);

   if usepanel
      set(handles.origin, 'visible', 'on');
   end

if 0
   %  Voxel Unit
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.5;

   pos = [x y w h];

   handles.Tcoord = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'left',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Axes Unit:');

   if usepanel
      set(handles.Tcoord, 'visible', 'on');
   end

   x = x + w + 0.005;
   w = info_pos(3)*0.5 - 0.005;

   pos = [x y w h];

   Font.FontSize   = 8;

   handles.coord = uicontrol('Parent',fig,'Style','popupmenu', ...
      'Units','Normalized', Font, ...
      'Position',pos, ...
      'BackgroundColor', [1 1 1], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'TooltipString','Choose Voxel or Millimeter',...
      'String',{'Voxel','Millimeter'},...
      'visible','off', ...
      'Callback','view_nii(''coordinates'');');

%      'TooltipString','Choose Voxel, MNI or Talairach Coordinates',...
%      'String',{'Voxel','MNI (mm)','Talairach (mm)'},...

   Font.FontSize   = 12;

   if usepanel
      set(handles.coord, 'visible', 'on');
   end
end

   %  Crosshair
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.4;

   pos = [x y w h];

   handles.Txhair = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'left',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Crosshair:');

   if usepanel
      set(handles.Txhair, 'visible', 'on');
   end

   x = info_pos(1) + info_pos(3)*0.5;
   w = info_pos(3)*0.2;
   h = inputline_space*0.7;

   pos = [x y w h];

   Font.FontSize   = 8;

   handles.xhair_color = uicontrol('Parent',fig,'Style','push', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'TooltipString','Crosshair Color',...
      'User',[1 0 0],...
      'String','Color',...
      'visible','off', ...
      'Callback','view_nii(''xhair_color'');');

   if usepanel
      set(handles.xhair_color, 'visible', 'on');
   end

   x = info_pos(1) + info_pos(3)*0.7;
   w = info_pos(3)*0.3;

   pos = [x y w h];

   handles.xhair = uicontrol('Parent',fig,'Style','popupmenu', ...
      'Units','Normalized', Font, ...
      'Position',pos, ...
      'BackgroundColor', [1 1 1], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'TooltipString','Display or Hide Crosshair',...
      'String',{'On','Off'},...
      'visible','off', ...
      'Callback','view_nii(''crosshair'');');

   if usepanel
      set(handles.xhair, 'visible', 'on');
   end

   %  Histogram & Color
   %
   x = info_pos(1);
   w = info_pos(3)*0.45;
   h = inputline_space * 1.5;

   pos = [x,  y+inputline_space*0.9,  w,  h];

   handles.hist_frame = uicontrol('Parent',fig, ...	
   	'Units','normal', ...
   	'BackgroundColor',[0.8 0.8 0.8], ...
   	'Position',pos, ...
        'visible','off', ...
   	'Style','frame');

   if usepanel
%      set(handles.hist_frame, 'visible', 'on');
   end

   handles.coord_frame = uicontrol('Parent',fig, ...	
   	'Units','normal', ...
   	'BackgroundColor',[0.8 0.8 0.8], ...
   	'Position',pos, ...
        'visible','off', ...
   	'Style','frame');

   if usepanel
      set(handles.coord_frame, 'visible', 'on');
   end

   x = info_pos(1) + info_pos(3)*0.475;
   w = info_pos(3)*0.525;
   h = inputline_space * 1.5;

   pos = [x,  y+inputline_space*0.9,  w,  h];

   handles.color_frame = uicontrol('Parent',fig, ...	
   	'Units','normal', ...
   	'BackgroundColor',[0.8 0.8 0.8], ...
   	'Position',pos, ...
        'visible','off', ...
   	'Style','frame');

   if usepanel
      set(handles.color_frame, 'visible', 'on');
   end

   x = info_pos(1) + info_pos(3)*0.025;
   y = y + inputline_space*1.2;
   w = info_pos(3)*0.2;
   h = inputline_space*0.7;

   pos = [x y w h];

   Font.FontSize   = 8;

   handles.hist_eq = uicontrol('Parent',fig,'Style','toggle', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'TooltipString','Histogram Equalization',...
      'String','Hist EQ',...
      'visible','off', ...
      'Callback','view_nii(''hist_eq'');');

   if usepanel
%      set(handles.hist_eq, 'visible', 'on');
   end

   x = x + w;
   w = info_pos(3)*0.2;

   pos = [x y w h];

   handles.hist_plot = uicontrol('Parent',fig,'Style','push', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'TooltipString','Histogram Plot',...
      'String','Hist Plot',...
      'visible','off', ...
      'Callback','view_nii(''hist_plot'');');

   if usepanel
%      set(handles.hist_plot, 'visible', 'on');
   end

   x = info_pos(1) + info_pos(3)*0.025;
   w = info_pos(3)*0.4;

   pos = [x y w h];

   handles.coord = uicontrol('Parent',fig,'Style','popupmenu', ...
      'Units','Normalized', Font, ...
      'Position',pos, ...
      'BackgroundColor', [1 1 1], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'TooltipString','Choose Voxel or Millimeter',...
      'String',{'Voxel','Millimeter'},...
      'visible','off', ...
      'Callback','view_nii(''coordinates'');');

%      'TooltipString','Choose Voxel, MNI or Talairach Coordinates',...
%      'String',{'Voxel','MNI (mm)','Talairach (mm)'},...

   if usepanel
      set(handles.coord, 'visible', 'on');
   end

   x = info_pos(1) + info_pos(3)*0.5;
   w = info_pos(3)*0.2;

   pos = [x y w h];

   handles.neg_color = uicontrol('Parent',fig,'Style','toggle', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'TooltipString','Negative Colormap',...
      'String','Negative',...
      'visible','off', ...
      'Callback','view_nii(''neg_color'');');

   if usepanel
      set(handles.neg_color, 'visible', 'on');
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      set(handles.neg_color, 'enable', 'off');
   end

   x = info_pos(1) + info_pos(3)*0.7;
   w = info_pos(3)*0.275;

   pos = [x y w h];

   handles.colorindex = uicontrol('Parent',fig,'Style','popupmenu', ...
      'Units','Normalized', Font, ...
      'Position',pos, ...
      'BackgroundColor', [1 1 1], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'TooltipString','Change Colormap',...
      'String',{'Custom','Bipolar','Gray','Jet','Cool','Bone','Hot','Copper','Pink'},...
      'value', colorindex, ...
      'visible','off', ...
      'Callback','view_nii(''color'');');

   if usepanel
      set(handles.colorindex, 'visible', 'on');
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      set(handles.colorindex, 'enable', 'off');
   end

   x = info_pos(1) + info_pos(3)*0.1;
   y = y + inputline_space;
   w = info_pos(3)*0.28;
   h = inputline_space*0.6;

   pos = [x y w h];

   Font.FontSize   = 8;

   handles.Thist = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Histogram');

   handles.Tcoord = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Axes Unit');

   if usepanel
%      set(handles.Thist, 'visible', 'on');
      set(handles.Tcoord, 'visible', 'on');
   end

   x = info_pos(1) + info_pos(3)*0.60;
   w = info_pos(3)*0.28;

   pos = [x y w h];

   handles.Tcolor = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Colormap');

   if usepanel
      set(handles.Tcolor, 'visible', 'on');
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      set(handles.Tcolor, 'enable', 'off');
   end

   %  Contrast Frame
   %
   x = info_pos(1);
   w = info_pos(3)*0.45;
   h = inputline_space * 2;

   pos = [x,  y+inputline_space*0.8,  w,  h];

   handles.contrast_frame = uicontrol('Parent',fig, ...	
   	'Units','normal', ...
   	'BackgroundColor',[0.8 0.8 0.8], ...
   	'Position',pos, ...
        'visible','off', ...
   	'Style','frame');

   if usepanel
      set(handles.contrast_frame, 'visible', 'on');
   end

   if colorindex < 2 | colorindex > 3
      set(handles.contrast_frame, 'visible', 'off');
   end

   %  Brightness Frame
   %
   x = info_pos(1) + info_pos(3)*0.475;
   w = info_pos(3)*0.525;

   pos = [x,  y+inputline_space*0.8,  w,  h];

   handles.brightness_frame = uicontrol('Parent',fig, ...	
   	'Units','normal', ...
   	'BackgroundColor',[0.8 0.8 0.8], ...
   	'Position',pos, ...
        'visible','off', ...
   	'Style','frame');

   if usepanel
      set(handles.brightness_frame, 'visible', 'on');
   end

   %  Contrast
   %
   x = info_pos(1) + info_pos(3)*0.025;
   y = y + inputline_space;
   w = info_pos(3)*0.4;
   h = inputline_space*0.6;

   pos = [x y w h];

   Font.FontSize   = 12;

   slider_step(1) = 5/255;
   slider_step(2) = 5.00001/255;

   handles.contrast = uicontrol('Parent',fig, ...
      'Style','slider','Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'left',...
      'BackgroundColor',[0.5 0.5 0.5],'ForegroundColor',[0 0 0],...
      'BusyAction','queue',...
      'TooltipString','Change contrast',...
      'Min',1,'Max',256,'SliderStep',slider_step, ...
      'Value',1, ...
      'visible','off', ...
      'Callback','view_nii(''contrast'');');

   if usepanel
      set(handles.contrast, 'visible', 'on');
   end

   if (nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511) & nii_view.numscan <= 1
      set(handles.contrast, 'enable', 'off');
   end

   if nii_view.numscan > 1
      set(handles.contrast, 'min', 1, 'max', nii_view.numscan, ...
         'sliderstep',[1/(nii_view.numscan-1) 1.00001/(nii_view.numscan-1)], ...
         'Callback', 'view_nii(''slider_change_scan'');');
   elseif colorindex < 2 | colorindex > 3
      set(handles.contrast, 'visible', 'off');
   elseif colorindex == 2
      set(handles.contrast,'value',128);
   end

   set(handles.contrast,'position',pos);	% linux66

   %  Brightness
   %
   x = info_pos(1) + info_pos(3)*0.5;
   w = info_pos(3)*0.475;

   pos = [x y w h];

   Font.FontSize   = 12;

   slider_step(1) = 1/50;
   slider_step(2) = 1.00001/50;

   handles.brightness = uicontrol('Parent',fig, ...
      'Style','slider','Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'left',...
      'BackgroundColor',[0.5 0.5 0.5],'ForegroundColor',[0 0 0],...
      'BusyAction','queue',...
      'TooltipString','Change brightness',...
      'Min',-1,'Max',1,'SliderStep',slider_step, ...
      'Value',0, ...
      'visible','off', ...
      'Callback','view_nii(''brightness'');');

   if usepanel
      set(handles.brightness, 'visible', 'on');
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      set(handles.brightness, 'enable', 'off');
   end

   set(handles.brightness,'position',pos);	% linux66

   %  Contrast text/def
   %
   x = info_pos(1) + info_pos(3)*0.025;
   y = y + inputline_space;
   w = info_pos(3)*0.22;

   pos = [x y w h];

   handles.Tcontrast = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'left',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Contrast:');

   if usepanel
      set(handles.Tcontrast, 'visible', 'on');
   end

   if (nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511) & nii_view.numscan <= 1
      set(handles.Tcontrast, 'enable', 'off');
   end

   if nii_view.numscan > 1
      set(handles.Tcontrast, 'string', 'Scan ID:');
      set(handles.contrast, 'TooltipString', 'Change Scan ID');
   elseif colorindex < 2 | colorindex > 3
      set(handles.Tcontrast, 'visible', 'off');
   end

   x = x + w;
   w = info_pos(3)*0.18;

   pos = [x y w h];

   Font.FontSize   = 8;

   handles.contrast_def = uicontrol('Parent',fig,'Style','push', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'TooltipString','Restore initial contrast',...
      'String','Reset',...
      'visible','off', ...
      'Callback','view_nii(''contrast_def'');');

   if usepanel
      set(handles.contrast_def, 'visible', 'on');
   end

   if (nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511) & nii_view.numscan <= 1
      set(handles.contrast_def, 'enable', 'off');
   end

   if nii_view.numscan > 1
      set(handles.contrast_def, 'style', 'edit', 'background', 'w', ...
         'TooltipString','Scan (or volume) index in the time series',...
         'string', '1', 'Callback', 'view_nii(''edit_change_scan'');');
   elseif colorindex < 2 | colorindex > 3
      set(handles.contrast_def, 'visible', 'off');
   end

   %  Brightness text/def
   %
   x = info_pos(1) + info_pos(3)*0.5;
   w = info_pos(3)*0.295;

   pos = [x y w h];

   Font.FontSize   = 12;

   handles.Tbrightness = uicontrol('Parent',fig,'Style','text', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'left',...
      'BackgroundColor', [0.8 0.8 0.8], 'ForegroundColor', [0 0 0],...
      'BusyAction','queue',...
      'visible','off', ...
      'String','Brightness:');

   if usepanel
      set(handles.Tbrightness, 'visible', 'on');
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      set(handles.Tbrightness, 'enable', 'off');
   end

   x = x + w;
   w = info_pos(3)*0.18;

   pos = [x y w h];

   Font.FontSize   = 8;

   handles.brightness_def = uicontrol('Parent',fig,'Style','push', ...
      'Units','Normalized', Font, ...
      'Position',pos, 'HorizontalAlignment', 'center',...
      'TooltipString','Restore initial brightness',...
      'String','Reset',...
      'visible','off', ...
      'Callback','view_nii(''brightness_def'');');

   if usepanel
      set(handles.brightness_def, 'visible', 'on');
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      set(handles.brightness_def, 'enable', 'off');
   end

   %  init image handles
   %
   handles.axial_image = [];
   handles.coronal_image = [];
   handles.sagittal_image = [];

   %  plot axial view
   %
   if ~isempty(nii_view.bgimg)
      bg_slice = squeeze(bgimg(:,:,nii_view.slices.axi));
      h1 = plot_view(fig, xdim, ydim, top_ax, bg_slice', clim, cbarminmax, ...
		handles, useimagesc, colorindex, color_map, ...
		colorlevel, highcolor, useinterp, nii_view.numscan);
      handles.axial_bg = h1;
   else
      handles.axial_bg = [];
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      img_slice = squeeze(nii.img(:,:,nii_view.slices.axi,:,setscanid));
      img_slice = permute(img_slice, [2 1 3]);
   else
      img_slice = squeeze(nii.img(:,:,nii_view.slices.axi,setscanid));
      img_slice = img_slice';
   end
   h1 = plot_view(fig, xdim, ydim, top_ax, img_slice, clim, cbarminmax, ...
	handles, useimagesc, colorindex, color_map, ...
	colorlevel, highcolor, useinterp, nii_view.numscan);
   set(h1,'buttondown','view_nii(''axial_image'');');
   handles.axial_image = h1;
   handles.axial_axes = top_ax;

   if size(img_slice,1) == 1 | size(img_slice,2) == 1
      set(top_ax,'visible','off');

      if isfield(handles,'sagittal_slider') & ishandle(handles.sagittal_slider)
         set(handles.sagittal_slider, 'visible', 'off');
      end

      if isfield(handles,'coronal_slider') & ishandle(handles.coronal_slider)
         set(handles.coronal_slider, 'visible', 'off');
      end

      if isfield(handles,'axial_slider') & ishandle(handles.axial_slider)
         set(handles.axial_slider, 'visible', 'off');
      end
   end

   %  plot coronal view
   %
   if ~isempty(nii_view.bgimg)
      bg_slice = squeeze(bgimg(:,nii_view.slices.cor,:));
      h1 = plot_view(fig, xdim, zdim, front_ax, bg_slice', clim, cbarminmax, ...
		handles, useimagesc, colorindex, color_map, ...
		colorlevel, highcolor, useinterp, nii_view.numscan);
      handles.coronal_bg = h1;
   else
      handles.coronal_bg = [];
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      img_slice = squeeze(nii.img(:,nii_view.slices.cor,:,:,setscanid));
      img_slice = permute(img_slice, [2 1 3]);
   else
      img_slice = squeeze(nii.img(:,nii_view.slices.cor,:,setscanid));
      img_slice = img_slice';
   end
   h1 = plot_view(fig, xdim, zdim, front_ax, img_slice, clim, cbarminmax, ...
	handles, useimagesc, colorindex, color_map, ...
	colorlevel, highcolor, useinterp, nii_view.numscan);
   set(h1,'buttondown','view_nii(''coronal_image'');');
   handles.coronal_image = h1;
   handles.coronal_axes = front_ax;

   if size(img_slice,1) == 1 | size(img_slice,2) == 1
      set(front_ax,'visible','off');

      if isfield(handles,'sagittal_slider') & ishandle(handles.sagittal_slider)
         set(handles.sagittal_slider, 'visible', 'off');
      end

      if isfield(handles,'coronal_slider') & ishandle(handles.coronal_slider)
         set(handles.coronal_slider, 'visible', 'off');
      end

      if isfield(handles,'axial_slider') & ishandle(handles.axial_slider)
         set(handles.axial_slider, 'visible', 'off');
      end
   end

   %  plot sagittal view
   %
   if ~isempty(nii_view.bgimg)
      bg_slice = squeeze(bgimg(nii_view.slices.sag,:,:));

      h1 = plot_view(fig, ydim, zdim, side_ax, bg_slice', clim, cbarminmax, ...
		handles, useimagesc, colorindex, color_map, ...
		colorlevel, highcolor, useinterp, nii_view.numscan);
      handles.sagittal_bg = h1;
   else
      handles.sagittal_bg = [];
   end

   if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
      img_slice = squeeze(nii.img(nii_view.slices.sag,:,:,:,setscanid));
      img_slice = permute(img_slice, [2 1 3]);
   else
      img_slice = squeeze(nii.img(nii_view.slices.sag,:,:,setscanid));
      img_slice = img_slice';
   end

   h1 = plot_view(fig, ydim, zdim, side_ax, img_slice, clim, cbarminmax, ...
	handles, useimagesc, colorindex, color_map, ...
	colorlevel, highcolor, useinterp, nii_view.numscan);
   set(h1,'buttondown','view_nii(''sagittal_image'');');
   set(side_ax,'Xdir', 'reverse');
   handles.sagittal_image = h1;
   handles.sagittal_axes = side_ax;

   if size(img_slice,1) == 1 | size(img_slice,2) == 1
      set(side_ax,'visible','off');

      if isfield(handles,'sagittal_slider') & ishandle(handles.sagittal_slider)
         set(handles.sagittal_slider, 'visible', 'off');
      end

      if isfield(handles,'coronal_slider') & ishandle(handles.coronal_slider)
         set(handles.coronal_slider, 'visible', 'off');
      end

      if isfield(handles,'axial_slider') & ishandle(handles.axial_slider)
         set(handles.axial_slider, 'visible', 'off');
      end
   end

   [top1_label, top2_label, side1_label, side2_label] = ...
	dir_label(fig, top_ax, front_ax, side_ax);

   %  store label handles
   %
   handles.top1_label = top1_label;
   handles.top2_label = top2_label;
   handles.side1_label = side1_label;
   handles.side2_label = side2_label;

   %  plot colorbar
   %
   if ~isempty(cbar_axes) & ~isempty(cbarminmax_axes)

if 0
      if isempty(color_map)
         level = colorlevel + num_highcolor;
      else
         level = size([color_map; highcolor], 1);
      end
end

      if isempty(color_map)
         level = colorlevel;
      else
         level = size([color_map], 1);
      end

      niiclass = class(nii.img);

      h1 = plot_cbar(fig, cbar_axes, cbarminmax_axes, cbarminmax, ...
		level, handles, useimagesc, colorindex, color_map, ...
		colorlevel, highcolor, niiclass, nii_view.numscan);
      handles.cbar_image = h1;
      handles.cbar_axes = cbar_axes;
      handles.cbarminmax_axes = cbarminmax_axes;

   end

   nii_view.handles = handles;		% store handles

   nii_view.usepanel = usepanel;		% whole panel at low right cornor
   nii_view.usestretch = usestretch;	% stretch display of voxel_size
   nii_view.useinterp = useinterp;	% use interpolation
   nii_view.colorindex = colorindex;	% store colorindex variable
   nii_view.buttondown = buttondown;	% command after button down click
   nii_view.cbarminmax = cbarminmax;	% store min max value for colorbar

   set_coordinates(nii_view,useinterp);	% coord unit

   if ~isfield(nii_view, 'axi_xhair') |  ...
      ~isfield(nii_view, 'cor_xhair') |  ...
      ~isfield(nii_view, 'sag_xhair')

      nii_view.axi_xhair = [];			% top cross hair
      nii_view.cor_xhair = [];			% front cross hair
      nii_view.sag_xhair = [];			% side cross hair

   end

   if ~isempty(color_map)
      nii_view.color_map = color_map;
   end

   if ~isempty(colorlevel)
      nii_view.colorlevel = colorlevel;
   end

   if ~isempty(highcolor)
      nii_view.highcolor = highcolor;
   end

   update_nii_view(nii_view);

   if ~isempty(setunit)
      update_unit(fig, setunit);
   end

   if ~isempty(setviewpoint)
      update_viewpoint(fig, setviewpoint);
   end

   if ~isempty(setcrosshaircolor)
      update_crosshaircolor(fig, setcrosshaircolor);
   end

   if ~isempty(usecrosshair)
      update_usecrosshair(fig, usecrosshair);
   end

   nii_menu = getappdata(fig, 'nii_menu');

   if ~isempty(nii_menu)
      if nii.hdr.dime.datatype == 128 | nii.hdr.dime.datatype == 511
         set(nii_menu.Minterp,'Userdata',1,'Label','Interp on','enable','off');
      elseif useinterp
         set(nii_menu.Minterp,'Userdata',0,'Label','Interp off');
      else
         set(nii_menu.Minterp,'Userdata',1,'Label','Interp on');
      end
   end

   windowbuttonmotion = get(fig, 'windowbuttonmotion');
   windowbuttonmotion = [windowbuttonmotion '; view_nii(''move_cursor'');'];
   set(fig, 'windowbuttonmotion', windowbuttonmotion);

   return;						% init


%----------------------------------------------------------------
function fig = update_img(img, fig, opt)

   nii_menu = getappdata(fig,'nii_menu');

   if ~isempty(nii_menu)
      set(nii_menu.Mzoom,'Userdata',1,'Label','Zoom on');   
      set(fig,'pointer','arrow');
      zoom off;
   end

   nii_view = getappdata(fig,'nii_view');
   change_interp = 0;

   if isfield(opt, 'useinterp') & opt.useinterp ~= nii_view.useinterp
      nii_view.useinterp = opt.useinterp;
      change_interp = 1;
   end

   setscanid = 1;

   if isfield(opt, 'setscanid')
      setscanid = round(opt.setscanid);

      if setscanid < 1
         setscanid = 1;
      end

      if setscanid > nii_view.numscan
         setscanid = nii_view.numscan;
      end
   end

   if isfield(opt, 'glblocminmax') & ~isempty(opt.glblocminmax)
      minvalue = opt.glblocminmax(1);
      maxvalue = opt.glblocminmax(2);
   else
      minvalue = img(:,:,:,setscanid);
      minvalue = double(minvalue(:));
      minvalue = min(minvalue(~isnan(minvalue)));
      maxvalue = img(:,:,:,setscanid);
      maxvalue = double(maxvalue(:));
      maxvalue = max(maxvalue(~isnan(maxvalue)));
   end

   if isfield(opt, 'setvalue')
      setvalue = opt.setvalue;

      if isfield(opt, 'glblocminmax') & ~isempty(opt.glblocminmax)
         minvalue = opt.glblocminmax(1);
         maxvalue = opt.glblocminmax(2);
      else
         minvalue = double(min(setvalue.val));
         maxvalue = double(max(setvalue.val));
      end

      bgimg = double(img);
      minbg = double(min(bgimg(:)));
      maxbg = double(max(bgimg(:)));

      bgimg = scale_in(bgimg, minbg, maxbg, 55) + 200;	% scale to 201~256

      cbarminmax = [minvalue maxvalue];

      if nii_view.useinterp

         %  scale signal data to 1~200
         %
         img = repmat(nan, size(img));
         img(setvalue.idx) = setvalue.val;

         %  200 level for source image
         %
         bgimg = single(scale_out(bgimg, cbarminmax(1), cbarminmax(2), 199));

      else

         bgimg(setvalue.idx) = NaN;
         minbg = double(min(bgimg(:)));
         maxbg = double(max(bgimg(:)));
         bgimg(setvalue.idx) = minbg;

         %  bgimg must be normalized to [201 256]
         %
         bgimg = 55 * (bgimg-min(bgimg(:))) / (max(bgimg(:))-min(bgimg(:))) + 201;
         bgimg(setvalue.idx) = 0;

         %  scale signal data to 1~200
         %
         img = zeros(size(img));
         img(setvalue.idx) = scale_in(setvalue.val, minvalue, maxvalue, 199);
         img = img + bgimg;
         bgimg = [];
         img = scale_out(img, cbarminmax(1), cbarminmax(2), 199);

         minvalue = double(min(img(:)));
         maxvalue = double(max(img(:)));

         if isfield(opt,'glblocminmax') & ~isempty(opt.glblocminmax)
            minvalue = opt.glblocminmax(1);
         end

      end

      nii_view.bgimg = bgimg;
      nii_view.setvalue = setvalue;

   else
      cbarminmax = [minvalue maxvalue];
   end

   update_cbarminmax(fig, cbarminmax);
   nii_view.cbarminmax = cbarminmax;
   nii_view.nii.img = img;
   nii_view.minvalue = minvalue;
   nii_view.maxvalue = maxvalue;
   nii_view.scanid = setscanid;
   change_colormap(fig);

   %  init color (gray) scaling to make sure the slice clim take the
   %  global clim [min(nii.img(:)) max(nii.img(:))]
   %
   if isempty(nii_view.bgimg)
      clim = [minvalue maxvalue];
   else
      clim = [minvalue double(max(nii_view.bgimg(:)))];
   end

   if clim(1) == clim(2)
      clim(2) = clim(1) + 0.000001;
   end

   if strcmpi(get(nii_view.handles.axial_image,'cdatamapping'), 'direct')
      useimagesc = 0;
   else
      useimagesc = 1;
   end

   if ~isempty(nii_view.bgimg)			% with interpolation

      Saxi = squeeze(nii_view.bgimg(:,:,nii_view.slices.axi));

      if isfield(nii_view.handles,'axial_bg') & ~isempty(nii_view.handles.axial_bg)
         set(nii_view.handles.axial_bg,'CData',double(Saxi)');
      else
         axes(nii_view.handles.axial_axes);

         if useimagesc
            nii_view.handles.axial_bg = surface(zeros(size(Saxi')),double(Saxi'),'edgecolor','none','facecolor','interp');
         else
            nii_view.handles.axial_bg = surface(zeros(size(Saxi')),double(Saxi'),'cdatamapping','direct','edgecolor','none','facecolor','interp');
         end

         order = get(gca,'child');
         order(find(order == nii_view.handles.axial_bg)) = [];
         order = [order; nii_view.handles.axial_bg];
         set(gca, 'child', order);
      end

   end

   if isfield(nii_view.handles,'axial_image'),
      if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
         Saxi = squeeze(nii_view.nii.img(:,:,nii_view.slices.axi,:,setscanid));
         Saxi = permute(Saxi, [2 1 3]);
      else
         Saxi = squeeze(nii_view.nii.img(:,:,nii_view.slices.axi,setscanid));
         Saxi = Saxi';
      end

      set(nii_view.handles.axial_image,'CData',double(Saxi));
   end

   set(nii_view.handles.axial_axes,'CLim',clim);

   if ~isempty(nii_view.bgimg)
      Scor = squeeze(nii_view.bgimg(:,nii_view.slices.cor,:));

      if isfield(nii_view.handles,'coronal_bg') & ~isempty(nii_view.handles.coronal_bg)
         set(nii_view.handles.coronal_bg,'CData',double(Scor)');
      else
         axes(nii_view.handles.coronal_axes);

         if useimagesc
            nii_view.handles.coronal_bg = surface(zeros(size(Scor')),double(Scor'),'edgecolor','none','facecolor','interp');
         else
            nii_view.handles.coronal_bg = surface(zeros(size(Scor')),double(Scor'),'cdatamapping','direct','edgecolor','none','facecolor','interp');
         end

         order = get(gca,'child');
         order(find(order == nii_view.handles.coronal_bg)) = [];
         order = [order; nii_view.handles.coronal_bg];
         set(gca, 'child', order);
      end
   end

   if isfield(nii_view.handles,'coronal_image'),
      if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
         Scor = squeeze(nii_view.nii.img(:,nii_view.slices.cor,:,:,setscanid));
         Scor = permute(Scor, [2 1 3]);
      else
         Scor = squeeze(nii_view.nii.img(:,nii_view.slices.cor,:,setscanid));
         Scor = Scor';
      end

      set(nii_view.handles.coronal_image,'CData',double(Scor));
   end

   set(nii_view.handles.coronal_axes,'CLim',clim);

   if ~isempty(nii_view.bgimg)
      Ssag = squeeze(nii_view.bgimg(nii_view.slices.sag,:,:));

      if isfield(nii_view.handles,'sagittal_bg') & ~isempty(nii_view.handles.sagittal_bg)
         set(nii_view.handles.sagittal_bg,'CData',double(Ssag)');
      else
         axes(nii_view.handles.sagittal_axes);

         if useimagesc
            nii_view.handles.sagittal_bg = surface(zeros(size(Ssag')),double(Ssag'),'edgecolor','none','facecolor','interp');
         else
            nii_view.handles.sagittal_bg = surface(zeros(size(Ssag')),double(Ssag'),'cdatamapping','direct','edgecolor','none','facecolor','interp');
         end

         order = get(gca,'child');
         order(find(order == nii_view.handles.sagittal_bg)) = [];
         order = [order; nii_view.handles.sagittal_bg];
         set(gca, 'child', order);
      end
   end

   if isfield(nii_view.handles,'sagittal_image'),
      if nii_view.nii.hdr.dime.datatype == 128 | nii_view.nii.hdr.dime.datatype == 511
         Ssag = squeeze(nii_view.nii.img(nii_view.slices.sag,:,:,:,setscanid));
         Ssag = permute(Ssag, [2 1 3]);
      else
         Ssag = squeeze(nii_view.nii.img(nii_view.slices.sag,:,:,setscanid));
         Ssag = Ssag';
      end

      set(nii_view.handles.sagittal_image,'CData',double(Ssag));
   end

   set(nii_view.handles.sagittal_axes,'CLim',clim);

   update_nii_view(nii_view);

   if isfield(opt, 'setvalue')

      if ~isfield(nii_view,'highcolor') | ~isequal(size(nii_view.highcolor),[56 3])

         %  55 level for brain structure (paded 0 for highcolor level 1, i.e. normal level 201, to make 56 highcolor)
         %
         update_highcolor(fig, [zeros(1,3);gray(55)], []);

      end

      if nii_view.colorindex ~= 2
         update_colorindex(fig, 2);
      end

      old_color = get(nii_view.handles.xhair_color,'user');

      if isequal(old_color, [1 0 0])
         update_crosshaircolor(fig, [1 1 0]);
      end

%      if change_interp
 %        update_useinterp(fig, nii_view.useinterp);
  %    end

   end

   if change_interp
      update_useinterp(fig, nii_view.useinterp);
   end

   return;						% update_img


%----------------------------------------------------------------
function [top_pos, front_pos, side_pos] = ...
			axes_pos(fig,area,vol_size,usestretch)

   set(fig,'unit','pixel');

   fig_pos = get(fig,'position');

   gap_x = 15/fig_pos(3);		% width of vertical scrollbar
   gap_y = 15/fig_pos(4);		% width of horizontal scrollbar

   a = (area(3) - gap_x * 1.3) * fig_pos(3) / (vol_size(1) + vol_size(2));	% no crosshair lost in zoom
   b = (area(4) - gap_y * 3) * fig_pos(4) / (vol_size(2) + vol_size(3));
   c = min([a b]);			% make sure 'ax' is inside 'area'

   top_w = vol_size(1) * c / fig_pos(3);
   side_w = vol_size(2) * c / fig_pos(3);
   top_h = vol_size(2) * c / fig_pos(4);
   side_h = vol_size(3) * c / fig_pos(4);
   side_x = area(1) + top_w + gap_x * 1.3;	% no crosshair lost in zoom
   side_y = area(2) + top_h + gap_y * 3;

   if usestretch
      if a > b				% top touched ceiling, use b
         d = (area(3) - gap_x * 1.3) / (top_w + side_w);	% no crosshair lost in zoom
         top_w = top_w * d;
         side_w = side_w * d;
         side_x = area(1) + top_w + gap_x * 1.3;	% no crosshair lost in zoom
      else
         d = (area(4) - gap_y * 3) / (top_h + side_h);
         top_h = top_h * d;
         side_h = side_h * d;
         side_y = area(2) + top_h + gap_y * 3;
      end
   end

   top_pos = [area(1) area(2)+gap_y top_w top_h];
   front_pos = [area(1) side_y top_w side_h];
   side_pos = [side_x side_y side_w side_h];

   set(fig,'unit','normal');

   return;						% axes_pos


%----------------------------------------------------------------
function [top_ax, front_ax, side_ax] ...
		= create_ax(fig, area, vol_size, usestretch)

   cur_fig = gcf;			% save h_wait fig
   figure(fig);

   [top_pos, front_pos, side_pos] = ...
			axes_pos(fig,area,vol_size,usestretch);

   nii_view = getappdata(fig, 'nii_view');

   if isempty(nii_view)
      top_ax = axes('position', top_pos);
      front_ax = axes('position', front_pos);
      side_ax = axes('position', side_pos);
   else
      top_ax = nii_view.handles.axial_axes;
      front_ax = nii_view.handles.coronal_axes;
      side_ax = nii_view.handles.sagittal_axes;

      set(top_ax, 'position', top_pos);
      set(front_ax, 'position', front_pos);
      set(side_ax, 'position', side_pos);
   end

   figure(cur_fig);

   return;						% create_ax


%----------------------------------------------------------------
function [cbar_axes, cbarminmax_axes] = create_cbar_axes(fig, cbar_area, nii_view)

   if isempty(cbar_area)		% without_cbar
      cbar_axes = [];
      cbarminmax_axes = [];
      return;
   end

   cur_fig = gcf;			% save h_wait fig
   figure(fig);

   if ~exist('nii_view', 'var')
      nii_view = getappdata(fig, 'nii_view');
   end

   if isempty(nii_view) | ~isfield(nii_view.handles,'cbar_axes') | isempty(nii_view.handles.cbar_axes)
      cbarminmax_axes = axes('position', cbar_area);
      cbar_axes = axes('position', cbar_area);
   else
      cbarminmax_axes = nii_view.handles.cbarminmax_axes;
      cbar_axes = nii_view.handles.cbar_axes;
      set(cbarminmax_axes, 'position', cbar_area);
      set(cbar_axes, 'position', cbar_area);
   end

   figure(cur_fig);

   return;					% create_cbar_axes


%----------------------------------------------------------------
function h1 = plot_view(fig, x, y, img_ax, img_slice, clim, ...
	cbarminmax, handles, useimagesc, colorindex, color_map, ...
	colorlevel, highcolor, useinterp, numscan)

   h1 = [];

   if x > 1 & y > 1,

      axes(img_ax);

      nii_view = getappdata(fig, 'nii_view');

      if isempty(nii_view)

         %  set colormap first
         %
         nii.handles = handles;
         nii.handles.axial_axes = img_ax;
         nii.colorindex = colorindex;
         nii.color_map = color_map;
         nii.colorlevel = colorlevel;
         nii.highcolor = highcolor;
         nii.numscan = numscan;

         change_colormap(fig, nii, colorindex, cbarminmax);

         if useinterp
            if useimagesc
               h1 = surface(zeros(size(img_slice)),double(img_slice),'edgecolor','none','facecolor','interp');
            else
               h1 = surface(zeros(size(img_slice)),double(img_slice),'cdatamapping','direct','edgecolor','none','facecolor','interp');
            end

            set(gca,'clim',clim);
         else
            if useimagesc
               h1 = imagesc(img_slice,clim);
            else
               h1 = image(img_slice);
            end

            set(gca,'clim',clim);
         end

      else

         h1 = nii_view.handles.axial_image;

         if ~isequal(get(h1,'parent'), img_ax)
            h1 = nii_view.handles.coronal_image;
         end

         if ~isequal(get(h1,'parent'), img_ax)
            h1 = nii_view.handles.sagittal_image;
         end

         set(h1, 'cdata', double(img_slice));
         set(h1, 'xdata', 1:size(img_slice,2));
         set(h1, 'ydata', 1:size(img_slice,1));

      end

      set(img_ax,'YDir','normal','XLimMode','manual','YLimMode','manual',...
         'ClimMode','manual','visible','off', ...
         'xtick',[],'ytick',[], 'clim', clim);

   end

   return;						% plot_view


%----------------------------------------------------------------
function h1 = plot_cbar(fig, cbar_axes, cbarminmax_axes, cbarminmax, ...
	level, handles, useimagesc, colorindex, color_map, ...
	colorlevel, highcolor, niiclass, numscan, nii_view)

   cbar_image = [1:level]';

   %  In a uint8 or uint16 indexed image, 0 points to the first row 
   %  in the colormap
   %
   if 0 % strcmpi(niiclass,'uint8') | strcmpi(niiclass,'uint16')
      % we use single for display anyway
      ylim = [0, level-1];
   else
      ylim = [1, level];
   end

   axes(cbarminmax_axes);

   plot([0 0], cbarminmax, 'w');
   axis tight;

   set(cbarminmax_axes,'YDir','normal', ...
      'XLimMode','manual','YLimMode','manual','YColor',[0 0 0], ...
      'XColor',[0 0 0],'xtick',[],'YAxisLocation','right');

   ylimb = get(cbarminmax_axes,'ylim');
   ytickb = get(cbarminmax_axes,'ytick');
   ytick=(ylim(2)-ylim(1))*(ytickb-ylimb(1))/(ylimb(2)-ylimb(1))+ylim(1);

   axes(cbar_axes);

   if ~exist('nii_view', 'var')
      nii_view = getappdata(fig, 'nii_view');
   end

   if isempty(nii_view) | ~isfield(nii_view.handles,'cbar_image') | isempty(nii_view.handles.cbar_image)

      %  set colormap first
      %
      nii.handles = handles;
      nii.colorindex = colorindex;
      nii.color_map = color_map;
      nii.colorlevel = colorlevel;
      nii.highcolor = highcolor;
      nii.numscan = numscan;

      change_colormap(fig, nii, colorindex, cbarminmax);
      h1 = image([0,1], [ylim(1),ylim(2)], cbar_image);

   else
      h1 = nii_view.handles.cbar_image;
      set(h1, 'cdata', double(cbar_image));
   end

   set(cbar_axes,'YDir','normal','XLimMode','manual', ...
	'YLimMode','manual','YColor',[0 0 0],'XColor',[0 0 0],'xtick',[], ...
	'YAxisLocation','right','ylim',ylim,'ytick',ytick,'yticklabel','');

   return;						% plot_cbar


%----------------------------------------------------------------
function set_coordinates(nii_view,useinterp)

    imgPlim.vox = nii_view.dims;
    imgNlim.vox = [1 1 1];

    if useinterp    
       xdata_ax = [imgNlim.vox(1) imgPlim.vox(1)];
       ydata_ax = [imgNlim.vox(2) imgPlim.vox(2)];
       zdata_ax = [imgNlim.vox(3) imgPlim.vox(3)];
    else
       xdata_ax = [imgNlim.vox(1)-0.5 imgPlim.vox(1)+0.5];
       ydata_ax = [imgNlim.vox(2)-0.5 imgPlim.vox(2)+0.5];
       zdata_ax = [imgNlim.vox(3)-0.5 imgPlim.vox(3)+0.5];
    end

    if isfield(nii_view.handles,'axial_image') & ~isempty(nii_view.handles.axial_image)
        set(nii_view.handles.axial_axes,'Xlim',xdata_ax);
        set(nii_view.handles.axial_axes,'Ylim',ydata_ax);
    end;
    if isfield(nii_view.handles,'coronal_image') & ~isempty(nii_view.handles.coronal_image)
        set(nii_view.handles.coronal_axes,'Xlim',xdata_ax);
        set(nii_view.handles.coronal_axes,'Ylim',zdata_ax);
    end;
    if isfield(nii_view.handles,'sagittal_image') & ~isempty(nii_view.handles.sagittal_image)
        set(nii_view.handles.sagittal_axes,'Xlim',ydata_ax);
        set(nii_view.handles.sagittal_axes,'Ylim',zdata_ax);
    end;

    return						% set_coordinates


%----------------------------------------------------------------
function set_image_value(nii_view),

    %  get coordinates of selected voxel and the image intensity there
    %
    sag = round(nii_view.slices.sag);
    cor = round(nii_view.slices.cor);
    axi = round(nii_view.slices.axi);

    if 0 % isfield(nii_view, 'disp')
       img = nii_view.disp;    
    else
       img = nii_view.nii.img;
    end

    if nii_view.nii.hdr.dime.datatype == 128
       imgvalue = [double(img(sag,cor,axi,1,nii_view.scanid)) double(img(sag,cor,axi,2,nii_view.scanid)) double(img(sag,cor,axi,3,nii_view.scanid))];
       set(nii_view.handles.imval,'Value',imgvalue);
       set(nii_view.handles.imval,'String',sprintf('%7.4g %7.4g %7.4g',imgvalue));
    elseif nii_view.nii.hdr.dime.datatype == 511
       R = double(img(sag,cor,axi,1,nii_view.scanid)) * (nii_view.nii.hdr.dime.glmax - ...
		nii_view.nii.hdr.dime.glmin) + nii_view.nii.hdr.dime.glmin;
       G = double(img(sag,cor,axi,2,nii_view.scanid)) * (nii_view.nii.hdr.dime.glmax - ...
		nii_view.nii.hdr.dime.glmin) + nii_view.nii.hdr.dime.glmin;
       B = double(img(sag,cor,axi,3,nii_view.scanid)) * (nii_view.nii.hdr.dime.glmax - ...
		nii_view.nii.hdr.dime.glmin) + nii_view.nii.hdr.dime.glmin;
       imgvalue = [double(img(sag,cor,axi,1,nii_view.scanid)) double(img(sag,cor,axi,2,nii_view.scanid)) double(img(sag,cor,axi,3,nii_view.scanid))];
       set(nii_view.handles.imval,'Value',imgvalue);
       imgvalue = [R G B];
       set(nii_view.handles.imval,'String',sprintf('%7.4g %7.4g %7.4g',imgvalue));
    else
       imgvalue = double(img(sag,cor,axi,nii_view.scanid));
       set(nii_view.handles.imval,'Value',imgvalue);

       if isnan(imgvalue) | imgvalue > nii_view.cbarminmax(2)
          imgvalue = 0;
       end

       set(nii_view.handles.imval,'String',sprintf('%.6g',imgvalue));
    end

    % Now update the coordinates of the selected voxel

    nii_view = update_imgXYZ(nii_view);

    if get(nii_view.handles.coord,'value') == 1,
       sag = nii_view.imgXYZ.vox(1);
       cor = nii_view.imgXYZ.vox(2);
       axi = nii_view.imgXYZ.vox(3);
       org = nii_view.origin;
    elseif get(nii_view.handles.coord,'value') == 2,
       sag = nii_view.imgXYZ.mm(1);
       cor = nii_view.imgXYZ.mm(2);
       axi = nii_view.imgXYZ.mm(3);
       org = [0 0 0];
    elseif get(nii_view.handles.coord,'value') == 3,
       sag = nii_view.imgXYZ.tal(1);
       cor = nii_view.imgXYZ.tal(2);
       axi = nii_view.imgXYZ.tal(3);
       org = [0 0 0];
    end

    set(nii_view.handles.impos,'Value',[sag,cor,axi]);

    if get(nii_view.handles.coord,'value') == 1,
        string = sprintf('%7.0f %7.0f %7.0f',sag,cor,axi);
        org_str = sprintf('%7.0f %7.0f %7.0f', org(1), org(2), org(3));
    else
        string = sprintf('%7.1f %7.1f %7.1f',sag,cor,axi);
        org_str = sprintf('%7.1f %7.1f %7.1f', org(1), org(2), org(3));
    end;
    
    set(nii_view.handles.impos,'String',string);
    set(nii_view.handles.origin, 'string', org_str);

    return						% set_image_value


%----------------------------------------------------------------
function nii_view = get_slice_position(nii_view,view),

    %  obtain slices that is in correct unit, then update slices
    %
    slices = nii_view.slices;
	
    switch view,
    case 'sag',
        currentpoint = get(nii_view.handles.sagittal_axes,'CurrentPoint');
        slices.cor = currentpoint(1,1);
        slices.axi = currentpoint(1,2);
    case 'cor',
        currentpoint = get(nii_view.handles.coronal_axes,'CurrentPoint');
        slices.sag = currentpoint(1,1);
        slices.axi = currentpoint(1,2);
    case 'axi',
        currentpoint = get(nii_view.handles.axial_axes,'CurrentPoint');
        slices.sag = currentpoint(1,1);
        slices.cor = currentpoint(1,2);
    end

    %  update nii_view.slices with the updated slices
    %    
    nii_view.slices.axi = round(slices.axi);
    nii_view.slices.cor = round(slices.cor);
    nii_view.slices.sag = round(slices.sag);

    return						% get_slice_position


%----------------------------------------------------------------
function nii_view = get_slider_position(nii_view),

    [nii_view.slices.sag,nii_view.slices.cor,nii_view.slices.axi] = deal(0);
    
    if isfield(nii_view.handles,'sagittal_slider'),
        if ishandle(nii_view.handles.sagittal_slider),
            nii_view.slices.sag = ...
		round(get(nii_view.handles.sagittal_slider,'Value'));
        end
    end
    
    if isfield(nii_view.handles,'coronal_slider'),
        if ishandle(nii_view.handles.coronal_slider),
            nii_view.slices.cor = ...
		round(nii_view.dims(2) - ...
		get(nii_view.handles.coronal_slider,'Value') + 1);
        end
    end
    
    if isfield(nii_view.handles,'axial_slider'),
        if ishandle(nii_view.handles.axial_slider),
            nii_view.slices.axi = ...
		round(get(nii_view.handles.axial_slider,'Value'));
        end
    end

    nii_view = check_slices(nii_view);

    return						% get_slider_position


%----------------------------------------------------------------
function nii_view = update_imgXYZ(nii_view),

   nii_view.imgXYZ.vox = ...
	[nii_view.slices.sag,nii_view.slices.cor,nii_view.slices.axi];
   nii_view.imgXYZ.mm = ...
	(nii_view.imgXYZ.vox - nii_view.origin) .* nii_view.voxel_size;
%   nii_view.imgXYZ.tal = mni2tal(nii_view.imgXYZ.mni);

   return						% update_imgXYZ


%----------------------------------------------------------------
function nii_view = convert2voxel(nii_view,slices),

    if get(nii_view.handles.coord,'value') == 1,

        %  [slices.axi, slices.cor, slices.sag] are in vox
        %
        nii_view.slices.axi = round(slices.axi);
        nii_view.slices.cor = round(slices.cor);
        nii_view.slices.sag = round(slices.sag);

    elseif get(nii_view.handles.coord,'value') == 2,

        %  [slices.axi, slices.cor, slices.sag] are in mm
        %
        xpix = nii_view.voxel_size(1);
        ypix = nii_view.voxel_size(2);
        zpix = nii_view.voxel_size(3);

        nii_view.slices.axi = round(slices.axi / zpix + nii_view.origin(3));
        nii_view.slices.cor = round(slices.cor / ypix + nii_view.origin(2));
        nii_view.slices.sag = round(slices.sag / xpix + nii_view.origin(1));
    elseif get(nii_view.handles.coord,'value') == 3,

        %  [slices.axi, slices.cor, slices.sag] are in talairach
        %
        xpix = nii_view.voxel_size(1);
        ypix = nii_view.voxel_size(2);
        zpix = nii_view.voxel_size(3);

        xyz_tal = [slices.sag, slices.cor, slices.axi];
        xyz_mni = tal2mni(xyz_tal);

        nii_view.slices.axi = round(xyz_mni(3) / zpix + nii_view.origin(3));
        nii_view.slices.cor = round(xyz_mni(2) / ypix + nii_view.origin(2));
        nii_view.slices.sag = round(xyz_mni(1) / xpix + nii_view.origin(1));

    end

    return						% convert2voxel


%----------------------------------------------------------------
function nii_view = check_slices(nii_view),

    img = nii_view.nii.img;
    
    [ SagSize, CorSize, AxiSize, TimeSize ] = size(img);
    if nii_view.slices.sag > SagSize, nii_view.slices.sag = SagSize; end;
    if nii_view.slices.sag < 1, nii_view.slices.sag = 1; end;
    if nii_view.slices.cor > CorSize, nii_view.slices.cor = CorSize; end;
    if nii_view.slices.cor < 1, nii_view.slices.cor = 1; end;
    if nii_view.slices.axi > AxiSize, nii_view.slices.axi = AxiSize; end;
    if nii_view.slices.axi < 1, nii_view.slices.axi = 1; end;
    if nii_view.scanid > TimeSize, nii_view.scanid = TimeSize; end;
    if nii_view.scanid < 1, nii_view.scanid = 1; end;
    
    return						% check_slices


%----------------------------------------------------------------
%
%  keep this function small, since it will be called for every click
%
function nii_view = update_nii_view(nii_view)

   %  add imgXYZ into nii_view struct
   %
   nii_view = check_slices(nii_view);
   nii_view = update_imgXYZ(nii_view);

   %  update xhair
   %
   p_axi = nii_view.imgXYZ.vox([1 2]);
   p_cor = nii_view.imgXYZ.vox([1 3]);
   p_sag = nii_view.imgXYZ.vox([2 3]);

   nii_view.axi_xhair = ...
	rri_xhair(p_axi, nii_view.axi_xhair, nii_view.handles.axial_axes);

   nii_view.cor_xhair = ...
	rri_xhair(p_cor, nii_view.cor_xhair, nii_view.handles.coronal_axes);

   nii_view.sag_xhair = ...
	rri_xhair(p_sag, nii_view.sag_xhair, nii_view.handles.sagittal_axes);

   setappdata(nii_view.fig, 'nii_view', nii_view);
   set_image_value(nii_view);

   return;						% update_nii_view


%----------------------------------------------------------------
function hist_plot(fig)

   nii_view = getappdata(fig,'nii_view');

   if isfield(nii_view, 'disp')
      img = nii_view.disp;    
   else
      img = nii_view.nii.img;
   end

   img = double(img(:));

   if length(unique(round(img))) == length(unique(img))
      is_integer = 1;
      range = max(img) - min(img) + 1;
      figure; hist(img, range);
      set(gca, 'xlim', [-range/5, max(img)]);
   else
      is_integer = 0;
      figure; hist(img);
   end

   xlabel('Voxel Intensity');
   ylabel('Voxel Numbers for Each Intensity');
   set(gcf, 'NumberTitle','off','Name','Histogram Plot');

   return;						% hist_plot


%----------------------------------------------------------------
function hist_eq(fig)

   nii_view = getappdata(fig,'nii_view');

   old_pointer = get(fig,'Pointer');
   set(fig,'Pointer','watch');

   if get(nii_view.handles.hist_eq,'value')
      max_img = double(max(nii_view.nii.img(:)));
      tmp = double(nii_view.nii.img) / max_img;		% normalize for histeq
      tmp = histeq(tmp(:));
      nii_view.disp = reshape(tmp, size(nii_view.nii.img));
      min_disp = min(nii_view.disp(:));
      nii_view.disp = (nii_view.disp - min_disp);		% range having eq hist
      nii_view.disp = nii_view.disp * max_img / max(nii_view.disp(:));
      nii_view.disp = single(nii_view.disp);
   else
      if isfield(nii_view, 'disp')
         nii_view.disp = nii_view.nii.img;
      else
         set(fig,'Pointer',old_pointer);
         return;
      end
   end

   %  update axial view
   %
   img_slice = squeeze(double(nii_view.disp(:,:,nii_view.slices.axi)));
   h1 = nii_view.handles.axial_image;
   set(h1, 'cdata', double(img_slice)');

   %  update coronal view
   %
   img_slice = squeeze(double(nii_view.disp(:,nii_view.slices.cor,:)));
   h1 = nii_view.handles.coronal_image;
   set(h1, 'cdata', double(img_slice)');

   %  update sagittal view
   %
   img_slice = squeeze(double(nii_view.disp(nii_view.slices.sag,:,:)));

   h1 = nii_view.handles.sagittal_image;
   set(h1, 'cdata', double(img_slice)');

   %  remove disp field if un-check 'histeq' button
   %
   if ~get(nii_view.handles.hist_eq,'value') & isfield(nii_view, 'disp')
      nii_view = rmfield(nii_view, 'disp');
   end

   update_nii_view(nii_view);

   set(fig,'Pointer',old_pointer);

   return;						% hist_eq


%----------------------------------------------------------------
function [top1_label, top2_label, side1_label, side2_label] = ...
		dir_label(fig, top_ax, front_ax, side_ax)

   nii_view = getappdata(fig,'nii_view');

   top_pos = get(top_ax,'position');
   front_pos = get(front_ax,'position');
   side_pos = get(side_ax,'position');

   top_gap_x = (side_pos(1)-top_pos(1)-top_pos(3)) / (2*top_pos(3));
   top_gap_y = (front_pos(2)-top_pos(2)-top_pos(4)) / (2*top_pos(4));
   side_gap_x = (side_pos(1)-top_pos(1)-top_pos(3)) / (2*side_pos(3));
   side_gap_y = (front_pos(2)-top_pos(2)-top_pos(4)) / (2*side_pos(4));

   top1_label_pos = [0, 1];			% rot0
   top2_label_pos = [1, 0];			% rot90
   side1_label_pos = [1, - side_gap_y];		% rot0
   side2_label_pos = [0, 0];			% rot90

   if isempty(nii_view)
      axes(top_ax);
      top1_label = text(double(top1_label_pos(1)),double(top1_label_pos(2)), ...
	'== X =>', ...
	'vertical', 'bottom', ...
	'unit', 'normal', 'fontsize', 8);

      axes(top_ax);
      top2_label = text(double(top2_label_pos(1)),double(top2_label_pos(2)), ...
	'== Y =>', ...
	'rotation', 90, 'vertical', 'top', ...
	'unit', 'normal', 'fontsize', 8);

      axes(side_ax);
      side1_label = text(double(side1_label_pos(1)),double(side1_label_pos(2)), ...
	'<= Y ==', ...
	'horizontal', 'right', 'vertical', 'top', ...
	'unit', 'normal', 'fontsize', 8);

      axes(side_ax);
      side2_label = text(double(side2_label_pos(1)),double(side2_label_pos(2)), ...
	'== Z =>', ...
	'rotation', 90, 'vertical', 'bottom', ...
	'unit', 'normal', 'fontsize', 8);
   else
      top1_label = nii_view.handles.top1_label;
      top2_label = nii_view.handles.top2_label;
      side1_label = nii_view.handles.side1_label;
      side2_label = nii_view.handles.side2_label;

      set(top1_label, 'position', [top1_label_pos 0]);
      set(top2_label, 'position', [top2_label_pos 0]);
      set(side1_label, 'position', [side1_label_pos 0]);
      set(side2_label, 'position', [side2_label_pos 0]);
   end

   return;						% dir_label


%----------------------------------------------------------------
function update_enable(h, opt);

   nii_view = getappdata(h,'nii_view');
   handles = nii_view.handles;

   if isfield(opt,'enablecursormove')
      if opt.enablecursormove
         v = 'on';
      else
         v = 'off';
      end

      set(handles.Timposcur, 'visible', v);
      set(handles.imposcur, 'visible', v);
      set(handles.Timvalcur, 'visible', v);
      set(handles.imvalcur, 'visible', v);
   end

   if isfield(opt,'enableviewpoint')
      if opt.enableviewpoint
         v = 'on';
      else
         v = 'off';
      end

      set(handles.Timpos, 'visible', v);
      set(handles.impos, 'visible', v);
      set(handles.Timval, 'visible', v);
      set(handles.imval, 'visible', v);
   end

   if isfield(opt,'enableorigin')
      if opt.enableorigin
         v = 'on';
      else
         v = 'off';
      end

      set(handles.Torigin, 'visible', v);
      set(handles.origin, 'visible', v);
   end

   if isfield(opt,'enableunit')
      if opt.enableunit
         v = 'on';
      else
         v = 'off';
      end

      set(handles.Tcoord, 'visible', v);
      set(handles.coord_frame, 'visible', v);
      set(handles.coord, 'visible', v);
   end

   if isfield(opt,'enablecrosshair')
      if opt.enablecrosshair
         v = 'on';
      else
         v = 'off';
      end

      set(handles.Txhair, 'visible', v);
      set(handles.xhair_color, 'visible', v);
      set(handles.xhair, 'visible', v);
   end

   if isfield(opt,'enablehistogram')
      if opt.enablehistogram
         v = 'on';
         vv = 'off';
      else
         v = 'off';
         vv = 'on';
      end

      set(handles.Tcoord, 'visible', vv);
      set(handles.coord_frame, 'visible', vv);
      set(handles.coord, 'visible', vv);

      set(handles.Thist, 'visible', v);
      set(handles.hist_frame, 'visible', v);
      set(handles.hist_eq, 'visible', v);
      set(handles.hist_plot, 'visible', v);
   end

   if isfield(opt,'enablecolormap')
      if opt.enablecolormap
         v = 'on';
      else
         v = 'off';
      end

      set(handles.Tcolor, 'visible', v);
      set(handles.color_frame, 'visible', v);
      set(handles.neg_color, 'visible', v);
      set(handles.colorindex, 'visible', v);
   end

   if isfield(opt,'enablecontrast')
      if opt.enablecontrast
         v = 'on';
      else
         v = 'off';
      end

      set(handles.Tcontrast, 'visible', v);
      set(handles.contrast_frame, 'visible', v);
      set(handles.contrast_def, 'visible', v);
      set(handles.contrast, 'visible', v);
   end

   if isfield(opt,'enablebrightness')
      if opt.enablebrightness
         v = 'on';
      else
         v = 'off';
      end

      set(handles.Tbrightness, 'visible', v);
      set(handles.brightness_frame, 'visible', v);
      set(handles.brightness_def, 'visible', v);
      set(handles.brightness, 'visible', v);
   end

   if isfield(opt,'enabledirlabel')
      if opt.enabledirlabel
         v = 'on';
      else
         v = 'off';
      end

      set(handles.top1_label, 'visible', v);
      set(handles.top2_label, 'visible', v);
      set(handles.side1_label, 'visible', v);
      set(handles.side2_label, 'visible', v);
   end

   if isfield(opt,'enableslider')
      if opt.enableslider
         v = 'on';
      else
         v = 'off';
      end

      if isfield(handles,'sagittal_slider') & ishandle(handles.sagittal_slider)
         set(handles.sagittal_slider, 'visible', v);
      end

      if isfield(handles,'coronal_slider') & ishandle(handles.coronal_slider)
         set(handles.coronal_slider, 'visible', v);
      end

      if isfield(handles,'axial_slider') & ishandle(handles.axial_slider)
         set(handles.axial_slider, 'visible', v);
      end
   end

   return;					% update_enable


%----------------------------------------------------------------
function update_usepanel(fig, usepanel)

   if isempty(usepanel)
      return;
   end

   if usepanel
      opt.enablecursormove = 1;
      opt.enableviewpoint = 1;
      opt.enableorigin = 1;
      opt.enableunit = 1;
      opt.enablecrosshair = 1;
%      opt.enablehistogram = 1;
      opt.enablecolormap = 1;
      opt.enablecontrast = 1;
      opt.enablebrightness = 1;
   else
      opt.enablecursormove = 0;
      opt.enableviewpoint = 0;
      opt.enableorigin = 0;
      opt.enableunit = 0;
      opt.enablecrosshair = 0;
%      opt.enablehistogram = 0;
      opt.enablecolormap = 0;
      opt.enablecontrast = 0;
      opt.enablebrightness = 0;
   end

   update_enable(fig, opt);

   nii_view = getappdata(fig,'nii_view');
   nii_view.usepanel = usepanel;
   setappdata(fig,'nii_view',nii_view);

   return;					% update_usepanel


%----------------------------------------------------------------
function update_usecrosshair(fig, usecrosshair)

   if isempty(usecrosshair)
      return;
   end

   if usecrosshair
      v=1;
   else
      v=2;
   end

   nii_view = getappdata(fig,'nii_view');
   set(nii_view.handles.xhair,'value',v);

   opt.command = 'crosshair';
   view_nii(fig, opt);

   return;					% update_usecrosshair


%----------------------------------------------------------------
function update_usestretch(fig, usestretch)

   nii_view = getappdata(fig,'nii_view');

   handles = nii_view.handles;
   fig = nii_view.fig;
   area = nii_view.area;
   vol_size = nii_view.voxel_size .* nii_view.dims;

   %  Three Axes & label
   %
   [top_ax, front_ax, side_ax] = ...
	create_ax(fig, area, vol_size, usestretch);

   dir_label(fig, top_ax, front_ax, side_ax);

   top_pos = get(top_ax,'position');
   front_pos = get(front_ax,'position');
   side_pos = get(side_ax,'position');

   %  Sagittal Slider
   %
   x = side_pos(1);
   y = top_pos(2) + top_pos(4);
   w = side_pos(3);
   h = (front_pos(2) - y) / 2;
   y = y + h;
   pos = [x y w h];

   if isfield(handles,'sagittal_slider') & ishandle(handles.sagittal_slider)
      set(handles.sagittal_slider,'position',pos);
   end

   %  Coronal Slider
   %
   x = top_pos(1);
   y = top_pos(2) + top_pos(4);
   w = top_pos(3);
   h = (front_pos(2) - y) / 2;
   y = y + h;
   pos = [x y w h];

   if isfield(handles,'coronal_slider') & ishandle(handles.coronal_slider)
      set(handles.coronal_slider,'position',pos);
   end

   %  Axial Slider
   %
   x = top_pos(1);
   y = area(2);
   w = top_pos(3);
   h = top_pos(2) - y;
   pos = [x y w h];

   if isfield(handles,'axial_slider') & ishandle(handles.axial_slider)
      set(handles.axial_slider,'position',pos);
   end

   %  plot info view
   %
%   info_pos = [side_pos([1,3]); top_pos([2,4])];
%   info_pos = info_pos(:);
   gap = side_pos(1)-(top_pos(1)+top_pos(3));
   info_pos(1) = side_pos(1) + gap;
   info_pos(2) = area(2);
   info_pos(3) = side_pos(3) - gap;
   info_pos(4) = top_pos(2) + top_pos(4) - area(2) - gap;

   num_inputline = 10;
   inputline_space =info_pos(4) / num_inputline;


   %  Image Intensity Value at Cursor
   %
   x = info_pos(1);
   y = info_pos(2);
   w = info_pos(3)*0.5;
   h = inputline_space*0.6;

   pos = [x y w h];
   set(handles.Timvalcur,'position',pos);

   x = x + w;
   w = info_pos(3)*0.5;

   pos = [x y w h];
   set(handles.imvalcur,'position',pos);

   %  Position at Cursor
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.5;

   pos = [x y w h];
   set(handles.Timposcur,'position',pos);

   x = x + w;
   w = info_pos(3)*0.5;

   pos = [x y w h];
   set(handles.imposcur,'position',pos);

   %  Image Intensity Value at Mouse Click
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.5;

   pos = [x y w h];
   set(handles.Timval,'position',pos);

   x = x + w;
   w = info_pos(3)*0.5;

   pos = [x y w h];
   set(handles.imval,'position',pos);

   %  Viewpoint Position at Mouse Click
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.5;

   pos = [x y w h];
   set(handles.Timpos,'position',pos);

   x = x + w + 0.005;
   y = y - 0.008;
   w = info_pos(3)*0.5;
   h = inputline_space*0.9;

   pos = [x y w h];
   set(handles.impos,'position',pos);

   %  Origin Position
   %
   x = info_pos(1);
   y = y + inputline_space*1.2;
   w = info_pos(3)*0.5;
   h = inputline_space*0.6;

   pos = [x y w h];
   set(handles.Torigin,'position',pos);

   x = x + w;
   w = info_pos(3)*0.5;

   pos = [x y w h];
   set(handles.origin,'position',pos);

if 0
   %  Axes Unit
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.5;

   pos = [x y w h];
   set(handles.Tcoord,'position',pos);

   x = x + w + 0.005;
   w = info_pos(3)*0.5 - 0.005;

   pos = [x y w h];
   set(handles.coord,'position',pos);
end

   %  Crosshair
   %
   x = info_pos(1);
   y = y + inputline_space;
   w = info_pos(3)*0.4;

   pos = [x y w h];
   set(handles.Txhair,'position',pos);

   x = info_pos(1) + info_pos(3)*0.5;
   w = info_pos(3)*0.2;
   h = inputline_space*0.7;

   pos = [x y w h];
   set(handles.xhair_color,'position',pos);

   x = info_pos(1) + info_pos(3)*0.7;
   w = info_pos(3)*0.3;

   pos = [x y w h];
   set(handles.xhair,'position',pos);

   %  Histogram & Color
   %
   x = info_pos(1);
   w = info_pos(3)*0.45;
   h = inputline_space * 1.5;
   pos = [x,  y+inputline_space*0.9,  w,  h];
   set(handles.hist_frame,'position',pos);
   set(handles.coord_frame,'position',pos);

   x = info_pos(1) + info_pos(3)*0.475;
   w = info_pos(3)*0.525;
   h = inputline_space * 1.5;

   pos = [x,  y+inputline_space*0.9,  w,  h];
   set(handles.color_frame,'position',pos);

   x = info_pos(1) + info_pos(3)*0.025;
   y = y + inputline_space*1.2;
   w = info_pos(3)*0.2;
   h = inputline_space*0.7;

   pos = [x y w h];
   set(handles.hist_eq,'position',pos);

   x = x + w;
   w = info_pos(3)*0.2;

   pos = [x y w h];
   set(handles.hist_plot,'position',pos);

   x = info_pos(1) + info_pos(3)*0.025;
   w = info_pos(3)*0.4;

   pos = [x y w h];
   set(handles.coord,'position',pos);

   x = info_pos(1) + info_pos(3)*0.5;
   w = info_pos(3)*0.2;
   pos = [x y w h];
   set(handles.neg_color,'position',pos);

   x = info_pos(1) + info_pos(3)*0.7;
   w = info_pos(3)*0.275;

   pos = [x y w h];
   set(handles.colorindex,'position',pos);

   x = info_pos(1) + info_pos(3)*0.1;
   y = y + inputline_space;
   w = info_pos(3)*0.28;
   h = inputline_space*0.6;

   pos = [x y w h];
   set(handles.Thist,'position',pos);
   set(handles.Tcoord,'position',pos);

   x = info_pos(1) + info_pos(3)*0.60;
   w = info_pos(3)*0.28;

   pos = [x y w h];
   set(handles.Tcolor,'position',pos);

   %  Contrast Frame
   %
   x = info_pos(1);
   w = info_pos(3)*0.45;
   h = inputline_space * 2;

   pos = [x,  y+inputline_space*0.8,  w,  h];
   set(handles.contrast_frame,'position',pos);

   %  Brightness Frame
   %
   x = info_pos(1) + info_pos(3)*0.475;
   w = info_pos(3)*0.525;

   pos = [x,  y+inputline_space*0.8,  w,  h];
   set(handles.brightness_frame,'position',pos);

   %  Contrast
   %
   x = info_pos(1) + info_pos(3)*0.025;
   y = y + inputline_space;
   w = info_pos(3)*0.4;
   h = inputline_space*0.6;

   pos = [x y w h];
   set(handles.contrast,'position',pos);

   %  Brightness
   %
   x = info_pos(1) + info_pos(3)*0.5;
   w = info_pos(3)*0.475;

   pos = [x y w h];
   set(handles.brightness,'position',pos);

   %  Contrast text/def
   %
   x = info_pos(1) + info_pos(3)*0.025;
   y = y + inputline_space;
   w = info_pos(3)*0.22;

   pos = [x y w h];
   set(handles.Tcontrast,'position',pos);

   x = x + w;
   w = info_pos(3)*0.18;

   pos = [x y w h];
   set(handles.contrast_def,'position',pos);

   %  Brightness text/def
   %
   x = info_pos(1) + info_pos(3)*0.5;
   w = info_pos(3)*0.295;

   pos = [x y w h];
   set(handles.Tbrightness,'position',pos);

   x = x + w;
   w = info_pos(3)*0.18;

   pos = [x y w h];
   set(handles.brightness_def,'position',pos);

   return;					% update_usestretch


%----------------------------------------------------------------
function update_useinterp(fig, useinterp)

   if isempty(useinterp)
      return;
   end

   nii_menu = getappdata(fig, 'nii_menu');

   if ~isempty(nii_menu)
      if get(nii_menu.Minterp,'user')
         set(nii_menu.Minterp,'Userdata',0,'Label','Interp off');
      else
         set(nii_menu.Minterp,'Userdata',1,'Label','Interp on');
      end
   end

   nii_view = getappdata(fig, 'nii_view');
   nii_view.useinterp = useinterp;

   if ~isempty(nii_view.handles.axial_image)
      if strcmpi(get(nii_view.handles.axial_image,'cdatamapping'), 'direct')
         useimagesc = 0;
      else
         useimagesc = 1;
      end
   elseif ~isempty(nii_view.handles.coronal_image)
      if strcmpi(get(nii_view.handles.coronal_image,'cdatamapping'), 'direct')
         useimagesc = 0;
      else
         useimagesc = 1;
      end
   else
      if strcmpi(get(nii_view.handles.sagittal_image,'cdatamapping'), 'direct')
         useimagesc = 0;
      else
         useimagesc = 1;
      end
   end

   if ~isempty(nii_view.handles.axial_image)
      img_slice = get(nii_view.handles.axial_image, 'cdata');
      delete(nii_view.handles.axial_image);
      axes(nii_view.handles.axial_axes);
      clim = get(gca,'clim');

      if useinterp
         if useimagesc
            nii_view.handles.axial_image = surface(zeros(size(img_slice)),double(img_slice),'edgecolor','none','facecolor','interp');
         else
            nii_view.handles.axial_image = surface(zeros(size(img_slice)),double(img_slice),'cdatamapping','direct','edgecolor','none','facecolor','interp');
         end
      else
         if useimagesc
            nii_view.handles.axial_image = imagesc('cdata',img_slice);
         else
            nii_view.handles.axial_image = image('cdata',img_slice);
         end
      end

      set(gca,'clim',clim);

      order = get(gca,'child');
      order(find(order == nii_view.handles.axial_image)) = [];
      order = [order; nii_view.handles.axial_image];

      if isfield(nii_view.handles,'axial_bg') & ~isempty(nii_view.handles.axial_bg)
         order(find(order == nii_view.handles.axial_bg)) = [];
         order = [order; nii_view.handles.axial_bg];
      end

      set(gca, 'child', order);

      if ~useinterp
         if isfield(nii_view.handles,'axial_bg') & ~isempty(nii_view.handles.axial_bg)
            delete(nii_view.handles.axial_bg);
            nii_view.handles.axial_bg = [];
         end
      end

      set(nii_view.handles.axial_image,'buttondown','view_nii(''axial_image'');');
   end

   if ~isempty(nii_view.handles.coronal_image)
      img_slice = get(nii_view.handles.coronal_image, 'cdata');
      delete(nii_view.handles.coronal_image);
      axes(nii_view.handles.coronal_axes);
      clim = get(gca,'clim');

      if useinterp
         if useimagesc
            nii_view.handles.coronal_image = surface(zeros(size(img_slice)),double(img_slice),'edgecolor','none','facecolor','interp');
         else
            nii_view.handles.coronal_image = surface(zeros(size(img_slice)),double(img_slice),'cdatamapping','direct','edgecolor','none','facecolor','interp');
         end
      else
         if useimagesc
            nii_view.handles.coronal_image = imagesc('cdata',img_slice);
         else
            nii_view.handles.coronal_image = image('cdata',img_slice);
         end
      end

      set(gca,'clim',clim);

      order = get(gca,'child');
      order(find(order == nii_view.handles.coronal_image)) = [];
      order = [order; nii_view.handles.coronal_image];

      if isfield(nii_view.handles,'coronal_bg') & ~isempty(nii_view.handles.coronal_bg)
         order(find(order == nii_view.handles.coronal_bg)) = [];
         order = [order; nii_view.handles.coronal_bg];
      end

      set(gca, 'child', order);

      if ~useinterp
         if isfield(nii_view.handles,'coronal_bg') & ~isempty(nii_view.handles.coronal_bg)
            delete(nii_view.handles.coronal_bg);
            nii_view.handles.coronal_bg = [];
         end
      end

      set(nii_view.handles.coronal_image,'buttondown','view_nii(''coronal_image'');');
   end

   if ~isempty(nii_view.handles.sagittal_image)
      img_slice = get(nii_view.handles.sagittal_image, 'cdata');
      delete(nii_view.handles.sagittal_image);
      axes(nii_view.handles.sagittal_axes);
      clim = get(gca,'clim');

      if useinterp
         if useimagesc
            nii_view.handles.sagittal_image = surface(zeros(size(img_slice)),double(img_slice),'edgecolor','none','facecolor','interp');
         else
            nii_view.handles.sagittal_image = surface(zeros(size(img_slice)),double(img_slice),'cdatamapping','direct','edgecolor','none','facecolor','interp');
         end
      else
         if useimagesc
            nii_view.handles.sagittal_image = imagesc('cdata',img_slice);
         else
            nii_view.handles.sagittal_image = image('cdata',img_slice);
         end
      end

      set(gca,'clim',clim);

      order = get(gca,'child');
      order(find(order == nii_view.handles.sagittal_image)) = [];
      order = [order; nii_view.handles.sagittal_image];

      if isfield(nii_view.handles,'sagittal_bg') & ~isempty(nii_view.handles.sagittal_bg)
         order(find(order == nii_view.handles.sagittal_bg)) = [];
         order = [order; nii_view.handles.sagittal_bg];
      end

      set(gca, 'child', order);

      if ~useinterp
         if isfield(nii_view.handles,'sagittal_bg') & ~isempty(nii_view.handles.sagittal_bg)
            delete(nii_view.handles.sagittal_bg);
            nii_view.handles.sagittal_bg = [];
         end
      end

      set(nii_view.handles.sagittal_image,'buttondown','view_nii(''sagittal_image'');');
   end

   if ~useinterp
      nii_view.bgimg = [];
   end

   set_coordinates(nii_view,useinterp);
   setappdata(fig, 'nii_view', nii_view);

   return;					% update_useinterp


%----------------------------------------------------------------
function update_useimagesc(fig, useimagesc)

   if isempty(useimagesc)
      return;
   end

   if useimagesc
      v='scaled';
   else
      v='direct';
   end

   nii_view = getappdata(fig,'nii_view');
   handles = nii_view.handles;

   if isfield(handles,'cbar_image') & ishandle(handles.cbar_image)
%      set(handles.cbar_image,'cdatamapping',v);
   end

   set(handles.axial_image,'cdatamapping',v);
   set(handles.coronal_image,'cdatamapping',v);
   set(handles.sagittal_image,'cdatamapping',v);

   return;					% update_useimagesc


%----------------------------------------------------------------
function update_shape(fig, area, usecolorbar, usestretch, useimagesc)

   nii_view = getappdata(fig,'nii_view');

   if isempty(usestretch)		% no change, get usestretch
      stretchchange = 0;
      usestretch = nii_view.usestretch;
   else					% change, set usestretch
      stretchchange = 1;
      nii_view.usestretch = usestretch;
   end

   if isempty(area)			% no change, get area

      areachange = 0;
      area = nii_view.area;

   elseif ~isempty(nii_view.cbar_area)	% change, set area & cbar_area

      areachange = 1;
      cbar_area = area;
      cbar_area(1) = area(1) + area(3)*0.93;
      cbar_area(3) = area(3)*0.04;
      area(3) = area(3)*0.9;		% 90% used for main axes

      [cbar_axes cbarminmax_axes] = create_cbar_axes(fig, cbar_area);

      nii_view.area = area;
      nii_view.cbar_area = cbar_area;

   else					% change, set area only
      areachange = 1;
      nii_view.area = area;
   end

   %  Add colorbar
   %
   if ~isempty(usecolorbar) & usecolorbar & isempty(nii_view.cbar_area)

      colorbarchange = 1;

      cbar_area = area;
      cbar_area(1) = area(1) + area(3)*0.93;
      cbar_area(3) = area(3)*0.04;
      area(3) = area(3)*0.9;		% 90% used for main axes

      %  create axes for colorbar
      %
      [cbar_axes cbarminmax_axes] = create_cbar_axes(fig, cbar_area);

      nii_view.area = area;
      nii_view.cbar_area = cbar_area;

      %  useimagesc follows axial image
      %
      if isempty(useimagesc)
         if strcmpi(get(nii_view.handles.axial_image,'cdatamap'),'scaled')
            useimagesc = 1;
         else
            useimagesc = 0;
         end
      end

      if isfield(nii_view, 'highcolor') & ~isempty(highcolor)
         num_highcolor = size(nii_view.highcolor,1);
      else
         num_highcolor = 0;
      end

      if isfield(nii_view, 'colorlevel') & ~isempty(nii_view.colorlevel)
         colorlevel = nii_view.colorlevel;
      else
         colorlevel = 256 - num_highcolor;
      end

      if isfield(nii_view, 'color_map')
         color_map = nii_view.color_map;
      else
         color_map = [];
      end

      if isfield(nii_view, 'highcolor')
         highcolor = nii_view.highcolor;
      else
         highcolor = [];
      end

      %  plot colorbar
      %
if 0
      if isempty(color_map)
         level = colorlevel + num_highcolor;
      else
         level = size([color_map; highcolor], 1);
      end
end

      if isempty(color_map)
         level = colorlevel;
      else
         level = size([color_map], 1);
      end

      cbar_image = [1:level]';

      niiclass = class(nii_view.nii.img);

      h1 = plot_cbar(fig, cbar_axes, cbarminmax_axes, nii_view.cbarminmax, ...
		level, nii_view.handles, useimagesc, nii_view.colorindex, ...
		color_map, colorlevel, highcolor, niiclass, nii_view.numscan);
      nii_view.handles.cbar_image = h1;
      nii_view.handles.cbar_axes = cbar_axes;
      nii_view.handles.cbarminmax_axes = cbar_axes;

   %  remove colorbar
   %
   elseif ~isempty(usecolorbar) & ~usecolorbar & ~isempty(nii_view.cbar_area)

      colorbarchange = 1;

      area(3) = area(3) / 0.9;

      nii_view.area = area;
      nii_view.cbar_area = [];

      nii_view.handles = rmfield(nii_view.handles,'cbar_image');
      delete(nii_view.handles.cbarminmax_axes);
      nii_view.handles = rmfield(nii_view.handles,'cbarminmax_axes');
      delete(nii_view.handles.cbar_axes);
      nii_view.handles = rmfield(nii_view.handles,'cbar_axes');

   else
      colorbarchange = 0;
   end

   if colorbarchange | stretchchange | areachange
      setappdata(fig,'nii_view',nii_view);
      update_usestretch(fig, usestretch);
   end

   return;					% update_shape


%----------------------------------------------------------------
function update_unit(fig, setunit)

   if isempty(setunit)
      return;
   end

   if strcmpi(setunit,'mm') | strcmpi(setunit,'millimeter') | strcmpi(setunit,'mni')
      v = 2;
%   elseif strcmpi(setunit,'tal') | strcmpi(setunit,'talairach')
 %     v = 3;
   elseif strcmpi(setunit,'vox') | strcmpi(setunit,'voxel')
      v = 1;
   else
      v = 1;
   end

   nii_view = getappdata(fig,'nii_view');
   set(nii_view.handles.coord, 'value', v);
   set_image_value(nii_view);

   return;					% update_unit


%----------------------------------------------------------------
function update_viewpoint(fig, setviewpoint)

   if isempty(setviewpoint)
      return;
   end

   nii_view = getappdata(fig,'nii_view');

   if length(setviewpoint) ~= 3
      error('Viewpoint position should contain [x y z]');
   end

   set(nii_view.handles.impos,'string',num2str(setviewpoint));

   opt.command = 'impos_edit';
   view_nii(fig, opt);

   set(nii_view.handles.axial_axes,'selected','on');
   set(nii_view.handles.axial_axes,'selected','off');
   set(nii_view.handles.coronal_axes,'selected','on');
   set(nii_view.handles.coronal_axes,'selected','off');
   set(nii_view.handles.sagittal_axes,'selected','on');
   set(nii_view.handles.sagittal_axes,'selected','off');

   return;					% update_viewpoint


%----------------------------------------------------------------
function update_scanid(fig, setscanid)

   if isempty(setscanid)
      return;
   end

   nii_view = getappdata(fig,'nii_view');

   if setscanid < 1
      setscanid = 1;
   end

   if setscanid > nii_view.numscan
      setscanid = nii_view.numscan;
   end

   set(nii_view.handles.contrast_def,'string',num2str(setscanid));
   set(nii_view.handles.contrast,'value',setscanid);

   opt.command = 'updateimg';
   opt.setscanid = setscanid;

   view_nii(fig, nii_view.nii.img, opt);

   return;					% update_scanid


%----------------------------------------------------------------
function update_crosshaircolor(fig, new_color)

   if isempty(new_color)
      return;
   end

   nii_view = getappdata(fig,'nii_view');
   xhair_color = nii_view.handles.xhair_color;

   set(xhair_color,'user',new_color);
   set(nii_view.axi_xhair.lx,'color',new_color);
   set(nii_view.axi_xhair.ly,'color',new_color);
   set(nii_view.cor_xhair.lx,'color',new_color);
   set(nii_view.cor_xhair.ly,'color',new_color);
   set(nii_view.sag_xhair.lx,'color',new_color);
   set(nii_view.sag_xhair.ly,'color',new_color);

   return;					% update_crosshaircolor


%----------------------------------------------------------------
function update_colorindex(fig, colorindex)

   if isempty(colorindex)
      return;
   end

   nii_view = getappdata(fig,'nii_view');
   nii_view.colorindex = colorindex;
   setappdata(fig, 'nii_view', nii_view);
   set(nii_view.handles.colorindex,'value',colorindex);

   opt.command = 'color';
   view_nii(fig, opt);

   return;					% update_colorindex


%----------------------------------------------------------------
function redraw_cbar(fig, colorlevel, color_map, highcolor)

   nii_view = getappdata(fig,'nii_view');

   if isempty(nii_view.cbar_area)
      return;
   end

   colorindex = nii_view.colorindex;

   if isempty(highcolor)
      num_highcolor = 0;
   else
      num_highcolor = size(highcolor,1);
   end

   if isempty(colorlevel)
      colorlevel=256;
   end

   if colorindex == 1
      colorlevel = size(color_map, 1);
   end

%   level = colorlevel + num_highcolor;
   level = colorlevel;

   cbar_image = [1:level]';

   cbar_area = nii_view.cbar_area;

   %  useimagesc follows axial image
   %
   if strcmpi(get(nii_view.handles.axial_image,'cdatamap'),'scaled')
      useimagesc = 1;
   else
      useimagesc = 0;
   end

   niiclass = class(nii_view.nii.img);

   delete(nii_view.handles.cbar_image);
   delete(nii_view.handles.cbar_axes);
   delete(nii_view.handles.cbarminmax_axes);

   [nii_view.handles.cbar_axes nii_view.handles.cbarminmax_axes] = ...
	create_cbar_axes(fig, cbar_area, []);

   nii_view.handles.cbar_image = plot_cbar(fig, ...
	nii_view.handles.cbar_axes, nii_view.handles.cbarminmax_axes, ...
	nii_view.cbarminmax, level, nii_view.handles, useimagesc, ...
	colorindex, color_map, colorlevel, highcolor, niiclass, ...
	nii_view.numscan, []);

   setappdata(fig, 'nii_view', nii_view);

   return;					% redraw_cbar


%----------------------------------------------------------------
function update_buttondown(fig, setbuttondown)

   if isempty(setbuttondown)
      return;
   end

   nii_view = getappdata(fig,'nii_view');
   nii_view.buttondown = setbuttondown;
   setappdata(fig, 'nii_view', nii_view);

   return;					% update_buttondown


%----------------------------------------------------------------
function update_cbarminmax(fig, cbarminmax)

   if isempty(cbarminmax)
      return;
   end

   nii_view = getappdata(fig, 'nii_view');

   if ~isfield(nii_view.handles, 'cbarminmax_axes')
      return;
   end

   nii_view.cbarminmax = cbarminmax;
   setappdata(fig, 'nii_view', nii_view);

   axes(nii_view.handles.cbarminmax_axes);

   plot([0 0], cbarminmax, 'w');
   axis tight;

   set(nii_view.handles.cbarminmax_axes,'YDir','normal', ...
      'XLimMode','manual','YLimMode','manual','YColor',[0 0 0], ...
      'XColor',[0 0 0],'xtick',[],'YAxisLocation','right');

   ylim = get(nii_view.handles.cbar_axes,'ylim');
   ylimb = get(nii_view.handles.cbarminmax_axes,'ylim');
   ytickb = get(nii_view.handles.cbarminmax_axes,'ytick');
   ytick=(ylim(2)-ylim(1))*(ytickb-ylimb(1))/(ylimb(2)-ylimb(1))+ylim(1);

   axes(nii_view.handles.cbar_axes);

   set(nii_view.handles.cbar_axes,'YDir','normal','XLimMode','manual', ...
	'YLimMode','manual','YColor',[0 0 0],'XColor',[0 0 0],'xtick',[], ...
	'YAxisLocation','right','ylim',ylim,'ytick',ytick,'yticklabel','');

   return;					% update_cbarminmax


%----------------------------------------------------------------
function update_highcolor(fig, highcolor, colorlevel)

   nii_view = getappdata(fig,'nii_view');

   if ischar(highcolor) & (isempty(colorlevel) | nii_view.colorindex == 1)
      return;
   end

   if ~ischar(highcolor)
      nii_view.highcolor = highcolor;

      if isempty(highcolor)
         nii_view = rmfield(nii_view, 'highcolor');
      end
   else
      highcolor = [];
   end

   if isempty(colorlevel) | nii_view.colorindex == 1
      nii_view.colorlevel = nii_view.colorlevel - size(highcolor,1);
   else
      nii_view.colorlevel = colorlevel;
   end

   setappdata(fig, 'nii_view', nii_view);

   if isfield(nii_view,'color_map')
      color_map = nii_view.color_map;
   else
      color_map = [];
   end

   redraw_cbar(fig, nii_view.colorlevel, color_map, highcolor);
   change_colormap(fig);

   return;					% update_highcolor


%----------------------------------------------------------------
function update_colormap(fig, color_map)

   if ischar(color_map)
      return;
   end

   nii_view = getappdata(fig,'nii_view');
   nii = nii_view.nii;
   minvalue = nii_view.minvalue;

   if isempty(color_map)
      if minvalue < 0
         colorindex = 2;
      else
         colorindex = 3;
      end

      nii_view = rmfield(nii_view, 'color_map');
      setappdata(fig,'nii_view',nii_view);
      update_colorindex(fig, colorindex);
      return;
   else
      colorindex = 1;
      nii_view.color_map = color_map;
      nii_view.colorindex = colorindex;
      setappdata(fig,'nii_view',nii_view);
      set(nii_view.handles.colorindex,'value',colorindex);
   end

   colorlevel = nii_view.colorlevel;

   if isfield(nii_view, 'highcolor')
      highcolor = nii_view.highcolor;
   else
      highcolor = [];
   end

   redraw_cbar(fig, colorlevel, color_map, highcolor);
   change_colormap(fig);

   opt.enablecontrast = 0;
   update_enable(fig, opt);

   return;					% update_colormap


%----------------------------------------------------------------
function status = get_status(h);

   nii_view = getappdata(h,'nii_view');

   status.fig = h;
   status.area = nii_view.area;

   if isempty(nii_view.cbar_area)
      status.usecolorbar = 0;
   else
      status.usecolorbar = 1;
      width = status.area(3) / 0.9;
      status.area(3) = width;
   end

   if strcmpi(get(nii_view.handles.imval,'visible'), 'on')
      status.usepanel = 1;
   else
      status.usepanel = 0;
   end

   if get(nii_view.handles.xhair,'value') == 1
      status.usecrosshair = 1;
   else
      status.usecrosshair = 0;
   end

   status.usestretch = nii_view.usestretch;

   if strcmpi(get(nii_view.handles.axial_image,'cdatamapping'), 'direct')
      status.useimagesc = 0;
   else
      status.useimagesc = 1;
   end

   status.useinterp = nii_view.useinterp;

   if get(nii_view.handles.coord,'value') == 1
      status.unit = 'vox';
   elseif get(nii_view.handles.coord,'value') == 2
      status.unit = 'mm';
   elseif get(nii_view.handles.coord,'value') == 3
      status.unit = 'tal';
   end

   status.viewpoint = get(nii_view.handles.impos,'value');
   status.scanid = nii_view.scanid;
   status.intensity = get(nii_view.handles.imval,'value');
   status.colorindex = get(nii_view.handles.colorindex,'value');

   if isfield(nii_view,'color_map')
      status.colormap = nii_view.color_map;
   else
      status.colormap = [];
   end

   status.colorlevel = nii_view.colorlevel;

   if isfield(nii_view,'highcolor')
      status.highcolor = nii_view.highcolor;
   else
      status.highcolor = [];
   end

   status.cbarminmax = nii_view.cbarminmax;
   status.buttondown = nii_view.buttondown;

   return;					% get_status


%----------------------------------------------------------------
function [custom_color_map, colorindex] ...
		 = change_colormap(fig, nii, colorindex, cbarminmax)

   custom_color_map = [];

   if ~exist('nii', 'var')
      nii_view = getappdata(fig,'nii_view');
   else
      nii_view = nii;
   end

   if ~exist('colorindex', 'var')
      colorindex = get(nii_view.handles.colorindex,'value');
   end

   if ~exist('cbarminmax', 'var')
      cbarminmax = nii_view.cbarminmax;
   end

   if isfield(nii_view, 'highcolor') & ~isempty(nii_view.highcolor)
      highcolor = nii_view.highcolor;
      num_highcolor = size(highcolor,1);
   else
      highcolor = [];
      num_highcolor = 0;
   end

%   if isfield(nii_view, 'colorlevel') & ~isempty(nii_view.colorlevel)
   if nii_view.colorlevel < 256
      num_color = nii_view.colorlevel;
   else
      num_color = 256 - num_highcolor;
   end

   contrast = [];

   if colorindex == 3					% for gray
      if nii_view.numscan > 1
         contrast = 1;
      else
         contrast = (num_color-1)*(get(nii_view.handles.contrast,'value')-1)/255+1;
         contrast = floor(contrast);
      end
   elseif colorindex == 2				% for bipolar
      if nii_view.numscan > 1
         contrast = 128;
      else
         contrast = get(nii_view.handles.contrast,'value');
      end
   end

   if isfield(nii_view,'color_map') & ~isempty(nii_view.color_map)
      color_map = nii_view.color_map;
      custom_color_map = color_map;
   elseif colorindex == 1
      [f p] = uigetfile('*.txt', 'Input colormap text file');

      if p==0
         colorindex = nii_view.colorindex;
         set(nii_view.handles.colorindex,'value',colorindex);
         return; 
      end;

      try
         custom_color_map = load(fullfile(p,f));
         loadfail = 0;
      catch
         loadfail = 1;
      end

      if loadfail | isempty(custom_color_map) | size(custom_color_map,2)~=3 ...
	| min(custom_color_map(:)) < 0 | max(custom_color_map(:)) > 1 

         msg = 'Colormap should be a Mx3 matrix with value between 0 and 1';
         msgbox(msg,'Error in colormap file');
         colorindex = nii_view.colorindex;
         set(nii_view.handles.colorindex,'value',colorindex);
         return;         
      end

      color_map = custom_color_map;
      nii_view.color_map = color_map;
   end

   switch colorindex
   case {2}
      color_map = bipolar(num_color, cbarminmax(1), cbarminmax(2), contrast);
   case {3}
      color_map = gray(num_color - contrast + 1);
   case {4}
      color_map = jet(num_color);
   case {5}
      color_map = cool(num_color);
   case {6}
      color_map = bone(num_color);
   case {7}
      color_map = hot(num_color);
   case {8}
      color_map = copper(num_color);
   case {9}
      color_map = pink(num_color);
   end

   nii_view.colorindex = colorindex;

   if ~exist('nii', 'var')
      setappdata(fig,'nii_view',nii_view);
   end

   if colorindex == 3
      color_map = [zeros(contrast,3); color_map(2:end,:)];
   end

   if get(nii_view.handles.neg_color,'value') & isempty(highcolor)
      color_map = flipud(color_map);
   elseif get(nii_view.handles.neg_color,'value') & ~isempty(highcolor)
      highcolor = flipud(highcolor);
   end

   brightness = get(nii_view.handles.brightness,'value');
   color_map = brighten(color_map, brightness);

   color_map = [color_map; highcolor];

   set(fig, 'colormap', color_map);

   return;					% change_colormap


%----------------------------------------------------------------
function move_cursor(fig)

   nii_view = getappdata(fig, 'nii_view');

   if isempty(nii_view)
      return;
   end

   axi = get(nii_view.handles.axial_axes, 'pos');
   cor = get(nii_view.handles.coronal_axes, 'pos');
   sag = get(nii_view.handles.sagittal_axes, 'pos');
   curr = get(fig, 'currentpoint');

   if		curr(1) >= axi(1) & curr(1) <= axi(1)+axi(3) & ...
		curr(2) >= axi(2) & curr(2) <= axi(2)+axi(4)

      curr = get(nii_view.handles.axial_axes, 'current');
      sag = curr(1,1);
      cor = curr(1,2);
      axi = nii_view.slices.axi;

   elseif	curr(1) >= cor(1) & curr(1) <= cor(1)+cor(3) & ...
		curr(2) >= cor(2) & curr(2) <= cor(2)+cor(4)

      curr = get(nii_view.handles.coronal_axes, 'current');
      sag = curr(1,1);
      cor = nii_view.slices.cor;
      axi = curr(1,2);

   elseif	curr(1) >= sag(1) & curr(1) <= sag(1)+sag(3) & ...
		curr(2) >= sag(2) & curr(2) <= sag(2)+sag(4)

      curr = get(nii_view.handles.sagittal_axes, 'current');

      sag = nii_view.slices.sag;
      cor = curr(1,1);
      axi = curr(1,2);

   else

      set(nii_view.handles.imvalcur,'String',' ');
      set(nii_view.handles.imposcur,'String',' ');
      return;

   end

    sag = round(sag);
    cor = round(cor);
    axi = round(axi);

    if sag < 1
       sag = 1;
    elseif sag > nii_view.dims(1)
       sag = nii_view.dims(1);
    end

    if cor < 1
       cor = 1;
    elseif cor > nii_view.dims(2)
       cor = nii_view.dims(2);
    end

    if axi < 1
       axi = 1;
    elseif axi > nii_view.dims(3)
       axi = nii_view.dims(3);
    end

    if 0 % isfield(nii_view, 'disp')
       img = nii_view.disp;    
    else
       img = nii_view.nii.img;
    end

    if nii_view.nii.hdr.dime.datatype == 128
       imgvalue = [double(img(sag,cor,axi,1,nii_view.scanid)) double(img(sag,cor,axi,2,nii_view.scanid)) double(img(sag,cor,axi,3,nii_view.scanid))];
       set(nii_view.handles.imvalcur,'String',sprintf('%7.4g %7.4g %7.4g',imgvalue));
    elseif nii_view.nii.hdr.dime.datatype == 511
       R = double(img(sag,cor,axi,1,nii_view.scanid)) * (nii_view.nii.hdr.dime.glmax - ...
		nii_view.nii.hdr.dime.glmin) + nii_view.nii.hdr.dime.glmin;
       G = double(img(sag,cor,axi,2,nii_view.scanid)) * (nii_view.nii.hdr.dime.glmax - ...
		nii_view.nii.hdr.dime.glmin) + nii_view.nii.hdr.dime.glmin;
       B = double(img(sag,cor,axi,3,nii_view.scanid)) * (nii_view.nii.hdr.dime.glmax - ...
		nii_view.nii.hdr.dime.glmin) + nii_view.nii.hdr.dime.glmin;
       imgvalue = [R G B];
       set(nii_view.handles.imvalcur,'String',sprintf('%7.4g %7.4g %7.4g',imgvalue));
    else
       imgvalue = double(img(sag,cor,axi,nii_view.scanid));

       if isnan(imgvalue) | imgvalue > nii_view.cbarminmax(2)
          imgvalue = 0;
       end

       set(nii_view.handles.imvalcur,'String',sprintf('%.6g',imgvalue));
    end

    nii_view.slices.sag = sag;
    nii_view.slices.cor = cor;
    nii_view.slices.axi = axi;

    nii_view = update_imgXYZ(nii_view);

    if get(nii_view.handles.coord,'value') == 1,
       sag = nii_view.imgXYZ.vox(1);
       cor = nii_view.imgXYZ.vox(2);
       axi = nii_view.imgXYZ.vox(3);
    elseif get(nii_view.handles.coord,'value') == 2,
       sag = nii_view.imgXYZ.mm(1);
       cor = nii_view.imgXYZ.mm(2);
       axi = nii_view.imgXYZ.mm(3);
    elseif get(nii_view.handles.coord,'value') == 3,
       sag = nii_view.imgXYZ.tal(1);
       cor = nii_view.imgXYZ.tal(2);
       axi = nii_view.imgXYZ.tal(3);
    end

    if get(nii_view.handles.coord,'value') == 1,
        string = sprintf('%7.0f %7.0f %7.0f',sag,cor,axi);
    else
        string = sprintf('%7.1f %7.1f %7.1f',sag,cor,axi);
    end;
    
    set(nii_view.handles.imposcur,'String',string);

    return;					% move_cursor


%----------------------------------------------------------------
function change_scan(hdl_str)

   fig = gcbf;
   nii_view = getappdata(fig,'nii_view');

   if strcmpi(hdl_str, 'edit_change_scan')		% edit
      hdl = nii_view.handles.contrast_def;
      setscanid = round(str2num(get(hdl, 'string')));
   else							% slider
      hdl = nii_view.handles.contrast;
      setscanid = round(get(hdl, 'value'));
   end

   update_scanid(fig, setscanid);

   return;					% change_scan


%----------------------------------------------------------------
function val = scale_in(val, minval, maxval, range)

   %  scale value into range
   %
   val = range*(double(val)-double(minval))/(double(maxval)-double(minval))+1;

   return;					% scale_in


%----------------------------------------------------------------
function val = scale_out(val, minval, maxval, range)

   %  according to [minval maxval] and range of color levels (e.g. 199)
   %  scale val back from any thing between 1~256 to a small number that
   %  is corresonding to [minval maxval].
   %
   val = (double(val)-1)*(double(maxval)-double(minval))/range+double(minval);

   return;					% scale_out

