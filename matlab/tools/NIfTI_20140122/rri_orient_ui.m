%  Return orientation of the current image:
%	orient is orientation 1x3 matrix, in that:
%	Three elements represent: [x y z]
%	Element value:	1 - Left to Right; 2 - Posterior to Anterior;
%			3 - Inferior to Superior; 4 - Right to Left;
%			5 - Anterior to Posterior; 6 - Superior to Inferior;
%  e.g.:
%	Standard RAS Orientation:	[1 2 3]
%	Standard RHOS Orientation:	[2 4 3]

%  Jimmy Shen (jimmy@rotman-baycrest.on.ca), 26-APR-04
%
function orient = rri_orient_ui(varargin)

   if nargin == 0
      init;
      orient_ui_fig = gcf;
      uiwait;					% wait for user finish

      orient = getappdata(gcf, 'orient');

      if isempty(orient)
         orient = [1 2 3];
      end

      if ishandle(orient_ui_fig)
         close(gcf);
      end

      return;
   end

   action = varargin{1};

   if strcmp(action, 'done')
      click_done;
   elseif strcmp(action, 'cancel')
      uiresume;
   end

   return;						% rri_orient_ui


%----------------------------------------------------------------------
function init

   save_setting_status = 'on';
   rri_orient_pos = [];

   try
      load('pls_profile');
   catch
   end

   try
      load('rri_pos_profile');
   catch
   end

   if ~isempty(rri_orient_pos) & strcmp(save_setting_status,'on')

      pos = rri_orient_pos;

   else

      w = 0.35;
      h = 0.4;
      x = (1-w)/2;
      y = (1-h)/2;

      pos = [x y w h];

   end

   handles.figure = figure('Color',[0.8 0.8 0.8], ...
        'Units','normal', ...
        'Name', 'Convert to standard RAS orientation', ...
        'NumberTitle','off', ...
        'MenuBar','none', ...
        'Position',pos, ...
        'WindowStyle', 'normal', ...
        'ToolBar','none');

   h0 = handles.figure;
   Font.FontUnits  = 'point';
   Font.FontSize   = 12;

   margin = .1;
   line_num = 6;
   line_ht = (1 - margin*2) / line_num;

   x = margin;
   y = 1 - margin - line_ht;
   w = 1 - margin * 2;
   h = line_ht * .7;

   pos = [x y w h];

   handles.Ttit = uicontrol('parent', h0, ...
	'style','text', ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','left',...
	'background', [0.8 0.8 0.8], ...
	'string', 'Please input orientation of the current image:');

   y = y - line_ht;
   w = .2;

   pos = [x y w h];

   handles.Tx_orient = uicontrol('parent', h0, ...
	'style','text', ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','left',...
	'background', [0.8 0.8 0.8], ...
	'string', 'X   Axes:');

   y = y - line_ht;

   pos = [x y w h];

   handles.Ty_orient = uicontrol('parent', h0, ...
	'style','text', ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','left',...
	'background', [0.8 0.8 0.8], ...
	'string', 'Y   Axes:');

   y = y - line_ht;

   pos = [x y w h];

   handles.Tz_orient = uicontrol('parent', h0, ...
	'style','text', ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','left',...
	'background', [0.8 0.8 0.8], ...
	'string', 'Z   Axes:');

   choice = {	'From Left to Right', 'From Posterior to Anterior', ...
		'From Inferior to Superior', 'From Right to Left', ...
		'From Anterior to Posterior', 'From Superior to Inferior'	};

   y = 1 - margin - line_ht;
   y = y - line_ht;
   w = 1 - margin - x - w;
   x = 1 - margin - w;

   pos = [x y w h];

   handles.x_orient = uicontrol('parent', h0, ...
	'style','popupmenu', ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','left',...
	'string', choice, ...
	'value', 1, ...
	'background', [1 1 1]);

   y = y - line_ht;

   pos = [x y w h];

   handles.y_orient = uicontrol('parent', h0, ...
	'style','popupmenu', ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','left',...
	'string', choice, ...
	'value', 2, ...
	'background', [1 1 1]);

   y = y - line_ht;

   pos = [x y w h];

   handles.z_orient = uicontrol('parent', h0, ...
	'style','popupmenu', ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','left',...
	'string', choice, ...
	'value', 3, ...
	'background', [1 1 1]);

   x = margin;
   y = y - line_ht * 1.5;
   w = .3;

   pos = [x y w h];

   handles.done = uicontrol('parent', h0, ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','center',...
        'callback', 'rri_orient_ui(''done'');', ...
	'string', 'Done');

   x = 1 - margin - w;

   pos = [x y w h];

   handles.cancel = uicontrol('parent', h0, ...
	'unit', 'normal', ...
	Font, ...
	'Position',pos, ...
	'HorizontalAlignment','center',...
        'callback', 'rri_orient_ui(''cancel'');', ...
	'string', 'Cancel');

   setappdata(h0, 'handles', handles);
   setappdata(h0, 'orient', [1 2 3]);

   return;						% init


%----------------------------------------------------------------------
function click_done

   handles = getappdata(gcf, 'handles');

   x_orient = get(handles.x_orient, 'value');
   y_orient = get(handles.y_orient, 'value');
   z_orient = get(handles.z_orient, 'value');

   orient = [x_orient y_orient z_orient];
   test_orient = [orient, orient + 3];
   test_orient = mod(test_orient, 3);

   if length(unique(test_orient)) ~= 3
      msgbox('Please don''t choose same or opposite direction','Error','modal');
      return;
   end

   setappdata(gcf, 'orient', [x_orient y_orient z_orient]);
   uiresume;

   return;						% click_done

