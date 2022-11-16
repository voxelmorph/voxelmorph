function [selected_file, selected_path] = rri_select_file(varargin)
%
%  USAGE: [selected_file, selected_path] = ...
%             rri_select_file(dir_name, fig_title)
%
%  Allow user to select a file from a list of Matlab competible
%	file format
%
%  Example:
%
%    [selected_file, selected_path] = ...
%		rri_select_file('/usr','Select Data File');
%
%  See Also RRI_GETFILES

%  -- Created June 2001 by Wilkin Chau, Rotman Research Institute
%
%  use rri_select_file to open & save Matlab recognized format
%  -- Modified Dec 2002 by Jimmy Shen, Rotman Research Institute
%

   if nargin == 0 | ischar(varargin{1}) 	% create rri_select_file figure

      dir_name = '';
      fig_title = 'Select a File';

      if nargin > 0
         dir_name = varargin{1};
      end

      if nargin > 1
         fig_title = varargin{2};
      end

      Init(fig_title,dir_name);
      uiwait;                           % wait for user finish

      selected_path = getappdata(gcf,'SelectedDirectory');
      selected_file = getappdata(gcf,'SelectedFile');

      cd (getappdata(gcf,'StartDirectory'));
      close(gcf);
      return;
   end;

   %  clear the message line,
   %
   h = findobj(gcf,'Tag','MessageLine');
   set(h,'String','');

   action = varargin{1}{1};

   %  change 'File format':
   %  update 'Files' & 'File selection' based on file pattern
   %
   if strcmp(action,'EditFilter'),
      EditFilter;

   %  run delete_fig when figure is closing
   %
   elseif strcmp(action,'delete_fig'),
      delete_fig;

   %  select 'Directories':
   %  go into the selected dir
   %  update 'Files' & 'File selection' based on file pattern
   %
   elseif strcmp(action,'select_dir'),
      select_dir;

   %  select 'Files':
   %  update 'File selection'
   %
   elseif strcmp(action,'select_file'),
      select_file;

   %  change 'File selection':
   %  if it is a file, select that,
   %  if it is more than a file (*), select those,
   %  if it is a directory, select based on file pattern
   %  
   elseif strcmp(action,'EditSelection'),
      EditSelection;

   %  clicked 'Select'
   %
   elseif strcmp(action,'DONE_BUTTON_PRESSED'),
      h = findobj(gcf,'Tag','SelectionEdit');
      [filepath,filename,fileext] = fileparts(get(h,'String'));

      if isempty(filepath) | isempty(filename) | isempty(fileext)
         setappdata(gcf,'SelectedDirectory',[]);
         setappdata(gcf,'SelectedFile',[]);
      else
         if ~strcmp(filepath(end),filesep)		% not end with filesep
            filepath = [filepath filesep];		% add a filesep to filepath
         end

         setappdata(gcf,'SelectedDirectory',filepath);
         setappdata(gcf,'SelectedFile',[filename fileext]);
      end

      if getappdata(gcf,'ready')			% ready to exit
         uiresume;
      end

   %  clicked 'cancel'
   %
   elseif strcmp(action,'CANCEL_BUTTON_PRESSED'),
      setappdata(gcf,'SelectedDirectory',[]);
      setappdata(gcf,'SelectedFile',[]);
      set(findobj(gcf,'Tag','FileList'),'String','');
      uiresume;
   end;

   return;


% --------------------------------------------------------------------
function Init(fig_title,dir_name),

   StartDirectory = pwd;
   if isempty(StartDirectory),
       StartDirectory = filesep;
   end;

   filter_disp = {'JPEG image (*.jpg)', ...
	'TIFF image, compressed (*.tif)', ...
	'EPS Level 1 (*.eps)', ...
	'Adobe Illustrator 88 (*.ai)', ...
	'Enhanced metafile (*.emf)', ...
	'Matlab Figure (*.fig)', ...
	'Matlab M-file (*.m)', ...
	'Portable bitmap (*.pbm)', ...
	'Paintbrush 24-bit (*.pcx)', ...
	'Portable Graymap (*.pgm)', ...
	'Portable Network Graphics (*.png)', ...
	'Portable Pixmap (*.ppm)', ...
   };

   filter_string = {'*.jpg', ...
	'*.tif', ...
	'*.eps', ...
	'*.ai', ...
	'*.emf', ...
	'*.fig', ...
	'*.m', ...
	'*.pbm', ...
	'*.pcx', ...
	'*.pgm', ...
	'*.png', ...
	'*.ppm', ...
   };

%   filter_disp = char(filter_disp);
   filter_string = char(filter_string);

   margine = 0.05;
   line_height = 0.07;
   char_height = line_height*0.8;

   save_setting_status = 'on';
   rri_select_file_pos = [];

   try
      load('pls_profile');
   catch
   end

   if ~isempty(rri_select_file_pos) & strcmp(save_setting_status,'on')

      pos = rri_select_file_pos;

   else

      w = 0.4;
      h = 0.6;
      x = (1-w)/2;
      y = (1-h)/2;

      pos = [x y w h];

   end

   h0 = figure('parent',0, 'Color',[0.8 0.8 0.8], ...
        'Units','normal', ...
        'Name',fig_title, ...
        'NumberTitle','off', ...
        'MenuBar','none', ...
        'Position', pos, ...
        'deleteFcn','rri_select_file({''delete_fig''});', ...
        'WindowStyle', 'modal', ...
        'Tag','GetFilesFigure', ...
        'ToolBar','none');

   x = margine;
   y = 1 - 1*line_height - margine;
   w = 1-2*x;
   h = char_height;

   pos = [x y w h];

   h1 = uicontrol('Parent',h0, ...		% Filter Label
        'Style','text', ...
        'Units','normal', ...
        'BackgroundColor',[0.8 0.8 0.8], ...
	'fontunit','normal', ...
        'FontSize',0.5, ...
        'HorizontalAlignment','left', ...
        'Position', pos, ...
        'String','Choose one of the file format:', ...
        'Tag','FilterLabel');

   y = 1 - 2*line_height - margine + line_height*0.2;
   w = 1-2*x;

   pos = [x y w h];

   h_filter = uicontrol('Parent',h0, ...	% Filter list
        'Style','popupmenu', ...
        'Units','normal', ...
        'BackgroundColor',[1 1 1], ...
	'fontunit','normal', ...
        'FontSize',0.5, ...
        'HorizontalAlignment','left', ...
        'Position', pos, ...
        'String', filter_disp, ...
        'user', filter_string, ...
	'value', 1, ...
        'Callback','rri_select_file({''EditFilter''});', ...
        'Tag','FilterEdit');

   y = 1 - 3*line_height - margine;
   w = 0.5 - x - margine/2;

   pos = [x y w h];

   h1 = uicontrol('Parent',h0, ...            % Directory Label
        'Style','text', ...
        'Units','normal', ...
        'BackgroundColor',[0.8 0.8 0.8], ...
	'fontunit','normal', ...
        'FontSize',0.5, ...
        'HorizontalAlignment','left', ...
        'ListboxTop',0, ...
        'Position', pos, ...
        'String','Directories', ...
        'Tag','DirectoryLabel');

   x = 0.5;
   y = 1 - 3*line_height - margine;
   w = 0.5 - margine;

   pos = [x y w h];

   h1 = uicontrol('Parent',h0, ...            % File Label
        'Style','text', ...
        'Units','normal', ...
        'BackgroundColor',[0.8 0.8 0.8], ...
	'fontunit','normal', ...
        'FontSize',0.5, ...
        'HorizontalAlignment','left', ...
        'ListboxTop',0, ...
        'Position', pos, ...
        'String','Files', ...
        'Tag','FileLabel');

   x = margine;
   y = 4*line_height + margine;
   w = 0.5 - x - margine/2;
   h = 1 - 7*line_height - 2*margine;

   pos = [x y w h];

   h_dir = uicontrol('Parent',h0, ...            % Directory Listbox
        'Style','listbox', ...
        'Units','normal', ...
	'fontunit','normal', ...
        'FontSize',0.08, ...
        'HorizontalAlignment','left', ...
        'Interruptible', 'off', ...
        'ListboxTop',1, ...
        'Position', pos, ...
        'String', '', ...
        'Callback','rri_select_file({''select_dir''});', ...
        'Tag','DirectoryList');

   x = 0.5;
   y = 4*line_height + margine;
   w = 0.5 - margine;
   h = 1 - 7*line_height - 2*margine;

   pos = [x y w h];

   h_file = uicontrol('Parent',h0, ...            % File Listbox
        'Style','listbox', ...
        'Units','normal', ...
	'fontunit','normal', ...
        'FontSize',0.08, ...
        'HorizontalAlignment','left', ...
        'ListboxTop',1, ...
        'Position', pos, ...
        'String', '', ...
        'Callback','rri_select_file({''select_file''});', ...
        'Tag','FileList');

   x = margine;
   y = 3*line_height + margine - line_height*0.2;
   w = 1-2*x;
   h = char_height;

   pos = [x y w h];

   h1 = uicontrol('Parent',h0, ...            % Selection Label
        'Style','text', ...
        'Units','normal', ...
        'BackgroundColor',[0.8 0.8 0.8], ...
        'fontunit','normal', ...
        'FontSize',0.5, ...
        'HorizontalAlignment','left', ...
        'Position', pos, ...
        'String','File you selected:', ...
        'Tag','SelectionLabel');

   y = 2*line_height + margine;
   w = 1-2*x;

   pos = [x y w h];

   h_select = uicontrol('Parent',h0, ...            % Selection Edit
        'Style','edit', ...
        'Units','normal', ...
        'BackgroundColor',[1 1 1], ...
        'fontunit','normal', ...
        'FontSize',0.5, ...
        'HorizontalAlignment','left', ...
        'Position', pos, ...
        'String', '', ...
        'Callback','rri_select_file({''EditSelection''});', ...
        'Tag','SelectionEdit');

   x = 2*margine;
   y = line_height/2 + margine;
   w = 0.2;
   h = line_height;

   pos = [x y w h];

   h_done = uicontrol('Parent',h0, ...                      % DONE
        'Units','normal', ...
        'fontunit','normal', ...
        'FontSize',0.5, ...
        'ListboxTop',0, ...
        'Position', pos, ...
        'HorizontalAlignment','center', ...
        'String','Save', ...			% 'Select', ...
        'Callback','rri_select_file({''DONE_BUTTON_PRESSED''});', ...
        'Tag','DONEButton');

   x = 1 - x - w;

   pos = [x y w h];

   h_cancel = uicontrol('Parent',h0, ...                      % CANCEL
        'Units','normal', ...
        'fontunit','normal', ...
        'FontSize',0.5, ...
        'ListboxTop',0, ...
        'Position', pos, ...
        'HorizontalAlignment','center', ...
        'String','Cancel', ...
        'Callback','rri_select_file({''CANCEL_BUTTON_PRESSED''});', ...
        'Tag','CANCELButton');

   if isempty(dir_name)
      dir_name = StartDirectory;
   end

   set(h_select,'string',dir_name);

   filter_select = get(h_filter,'value');
   filter_pattern = filter_string(filter_select,:);

   setappdata(gcf,'FilterPattern',deblank(filter_pattern));
   setappdata(gcf,'filter_string',filter_string);

   setappdata(gcf,'h_filter', h_filter);
   setappdata(gcf,'h_dir', h_dir);
   setappdata(gcf,'h_file', h_file);
   setappdata(gcf,'h_select', h_select);
   setappdata(gcf,'h_done', h_done);
   setappdata(gcf,'h_cancel', h_cancel);
   setappdata(gcf,'StartDirectory',StartDirectory);

   EditSelection;

   h_file = getappdata(gcf,'h_file');
   if isempty(get(h_file,'string'))
      setappdata(gcf,'ready',0);
   else
      setappdata(gcf,'ready',1);
   end

   return;					% Init


%  called by all the actions, to update 'Directories' or 'Files'
%  based on filter_pattern. Select first file in filelist.
%
% --------------------------------------------------------------------

function update_dirlist;

   filter_path = getappdata(gcf,'curr_dir');
   filter_pattern = getappdata(gcf,'FilterPattern');

   if exist(filter_pattern) == 2	% user input specific filename
      is_single_file = 1;		% need manually take path out later
   else
      is_single_file = 0;
   end

   % take the file path out from filter_pattern
   %
   [fpath fname fext] = fileparts(filter_pattern);
   filter_pattern = [fname fext];

   dir_struct = dir(filter_path);
   if isempty(dir_struct)
      msg = 'ERROR: Directory not found!';
      uiwait(msgbox(msg,'File Selection Error','modal'));
      return;
   end;

   old_pointer = get(gcf,'Pointer');
   set(gcf,'Pointer','watch');
   
   dir_list = dir_struct(find([dir_struct.isdir] == 1));
   [sorted_dir_names,sorted_dir_index] = sortrows({dir_list.name}');

   dir_struct = dir([filter_path filesep filter_pattern]);
   if isempty(dir_struct)
      sorted_file_names = [];
   else
      file_list = dir_struct(find([dir_struct.isdir] == 0));

      if is_single_file			% take out path
         tmp = file_list.name;
         [fpath fname fext] = fileparts(tmp);
         file_list.name = [fname fext];
      end

      [sorted_file_names,sorted_file_index] = sortrows({file_list.name}');
   end;

   disp_dir_names = [];			% if need full path, use this
					% instead of sorted_dir_names
   for i=1:length(sorted_dir_names)
      tmp = [filter_path filesep sorted_dir_names{i}];
      disp_dir_names = [disp_dir_names {tmp}];
   end

   h = findobj(gcf,'Tag','DirectoryList');
   set(h,'String',sorted_dir_names,'Value',1);

   h = findobj(gcf,'Tag','FileList');
   set(h,'String',sorted_file_names,'value',1);

   h_select = getappdata(gcf,'h_select');
   if strcmp(filter_path(end),filesep)		% filepath end with filesep
      filter_path = filter_path(1:end-1);	% take filesep out
   end

   if isempty(sorted_file_names)
      set(h_select,'string',[filter_path filesep]);
   else
      set(h_select,'string',[filter_path filesep sorted_file_names{1}]);
   end

   set(gcf,'Pointer',old_pointer);

   return; 					% update_dirlist


%  change 'File format':
%  update 'Files' & 'File selection' based on file pattern
%
% --------------------------------------------------------------------

function EditFilter()

   filter_select = get(gcbo,'value');
   filter_string = getappdata(gcf,'filter_string');
   filter_pattern = filter_string(filter_select,:);
   filter_path = getappdata(gcf,'curr_dir');

   % update filter_pattern
   setappdata(gcf,'FilterPattern',deblank(filter_pattern));

   if isempty(filter_path),
       filter_path = filesep;
   end;

   update_dirlist;

   h_file = getappdata(gcf,'h_file');
   if isempty(get(h_file,'string'))
      setappdata(gcf,'ready',0);
   else
      setappdata(gcf,'ready',1);
   end

   return;					% EditFilter


%  select 'Directories':
%  go into the selected dir
%  update 'Files' & 'File selection' based on file pattern
%
% --------------------------------------------------------------------

function select_dir()

   listed_dir = get(gcbo,'String');
   selected_dir_idx = get(gcbo,'Value');
   selected_dir = listed_dir{selected_dir_idx};
   curr_dir = getappdata(gcf,'curr_dir');
   
   %  update the selection box
   %
   try 
      cd ([curr_dir filesep selected_dir]);
   catch
      msg = 'ERROR: Cannot access directory';
      uiwait(msgbox(msg,'File Selection Error','modal'));
      return;
   end;

   if isempty(pwd)
      curr_dir = filesep;
   else
      curr_dir = pwd;
   end;

   setappdata(gcf,'curr_dir',curr_dir);
   update_dirlist;

   h_file = getappdata(gcf,'h_file');
   if isempty(get(h_file,'string'))
      setappdata(gcf,'ready',0);
   else
      setappdata(gcf,'ready',1);
   end

   return;					% select_dir


%  select 'Files':
%  update 'File selection'
%
% --------------------------------------------------------------------

function select_file()

   setappdata(gcf,'ready',1);
   listed_file = get(gcbo,'String');
   selected_file_idx = get(gcbo,'Value');
   selected_file = listed_file{selected_file_idx};
   curr_dir = getappdata(gcf,'curr_dir');

   if strcmp(curr_dir(end),filesep)		% filepath end with filesep
      curr_dir = curr_dir(1:end-1);	% take filesep out
   end

   h_select = getappdata(gcf,'h_select');
   set(h_select,'string',[curr_dir filesep selected_file]);

   return;					% select_file


%  change 'File selection':
%  if it is a file, select that,
%  if it is more than a file (*), select those,
%  if it is a directory, select based on file pattern
%  
% --------------------------------------------------------------------

function EditSelection()

   filter_string = getappdata(gcf,'filter_string');
   h_select = getappdata(gcf,'h_select');
   selected_file = get(h_select,'string');

   if exist(selected_file) == 7			% if user enter a dir
      setappdata(gcf,'ready',0);
      setappdata(gcf,'curr_dir',selected_file);		% get new dir
      update_dirlist;
   else

      setappdata(gcf,'ready',1);

      [fpath fname fext]= fileparts(selected_file);
      if exist(fpath) ~=7			% fpath is not a dir
         setappdata(gcf,'ready',0);
         msg = 'ERROR: Cannot access directory';
         uiwait(msgbox(msg,'File Selection Error','modal'));
      end

      %  if the file format user entered is not supported by matlab
      if isempty(strmatch(['*',fext],filter_string,'exact'))
         setappdata(gcf,'ready',0);
         msg = 'ERROR: File format is not supported by Matlab.';
         uiwait(msgbox(msg,'File Selection Error','modal'));
      end

   end

   return;					% EditSelection


% --------------------------------------------------------------------

function delete_fig()

   try
      load('pls_profile');
      pls_profile = which('pls_profile.mat');

      rri_select_file_pos = get(gcbf,'position');

      save(pls_profile, '-append', 'rri_select_file_pos');
   catch
   end

   return;

