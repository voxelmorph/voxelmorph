%  Imbed a file menu to any figure. If file menu exist, it will append
%  to the existing file menu. This file menu includes: Copy to clipboard,
%  print, save, close etc.
%
%  Usage: rri_file_menu(fig);
%
%         rri_file_menu(fig,0) means no 'Close' menu.
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
%--------------------------------------------------------------------

function rri_file_menu(action, varargin)

   if isnumeric(action)
      fig = action;
      action = 'init';
   end

   %  clear the message line,
   %
   h = findobj(gcf,'Tag','MessageLine');
   set(h,'String','');

   if ~strcmp(action, 'init')
      set(gcbf, 'InvertHardcopy','off');
%      set(gcbf, 'PaperPositionMode','auto');
   end

   switch action
      case {'init'}
         if nargin > 1
            init(fig, 1);		% no 'close' menu
         else
            init(fig, 0);
         end
      case {'print_fig'}
         printdlg(gcbf);
      case {'copy_fig'}
         copy_fig;
      case {'export_fig'}
         export_fig;
   end

   return					% rri_file_menu


%------------------------------------------------
%
%  Create (or append) File menu
%
function init(fig, no_close)

   %  search for file menu
   %
   h_file = [];
   menuitems = findobj(fig, 'type', 'uimenu');

   for i=1:length(menuitems)
      filelabel = get(menuitems(i),'label');

      if strcmpi(strrep(filelabel, '&', ''), 'file')
         h_file = menuitems(i);
         break;
      end
   end

   set(fig, 'menubar', 'none');

   if isempty(h_file)
      if isempty(menuitems)
         h_file = uimenu('parent', fig, 'label', 'File');
      else
         h_file = uimenu('parent', fig, 'label', 'Copy Figure');
      end

      h1 = uimenu('parent', h_file, ...
         'callback','rri_file_menu(''copy_fig'');', ...
         'label','Copy to Clipboard');
   else
      h1 = uimenu('parent', h_file, ...
         'callback','rri_file_menu(''copy_fig'');', ...
         'separator','on', ...
         'label','Copy to Clipboard');
   end

   h2 = uimenu(h_file, ...
      'callback','pagesetupdlg(gcbf);', ...
      'label','Page Setup...');

   h2 = uimenu(h_file, ...
      'callback','printpreview(gcbf);', ...
      'label','Print Preview...');

   h2 = uimenu('parent', h_file, ...
      'callback','printdlg(gcbf);', ...
      'label','Print Figure ...');

   h2 = uimenu('parent', h_file, ...
      'callback','rri_file_menu(''export_fig'');', ...
      'label','Save Figure ...');

   arch = computer;
   if ~strcmpi(arch(1:2),'PC')
      set(h1, 'enable', 'off');
   end

   if ~no_close
      h1 = uimenu('parent', h_file, ...
         'callback','close(gcbf);', ...
         'separator','on', ...
         'label','Close');
   end

   return;					% init


%------------------------------------------------
%
%  Copy to clipboard
%
function copy_fig

   arch = computer;
   if(~strcmpi(arch(1:2),'PC'))
      error('copy to clipboard can only be used under MS Windows');
      return;
   end

   print -noui -dbitmap;

   return					% copy_fig


%------------------------------------------------
%
%  Save as an image file
%
function export_fig

   curr = pwd;
   if isempty(curr)
      curr = filesep;
   end

   [selected_file, selected_path] = rri_select_file(curr,'Save As');

   if isempty(selected_file) | isempty(selected_path)
      return;
   end

   filename = [selected_path selected_file];

   if(exist(filename,'file')==2)		% file exist

      dlg_title = 'Confirm File Overwrite';
      msg = ['File ',filename,' exist. Are you sure you want to overwrite it?'];
      response = questdlg(msg,dlg_title,'Yes','No','Yes');

      if(strcmp(response,'No'))
         return;
      end

   end

   old_pointer = get(gcbf,'pointer');
   set(gcbf,'pointer','watch');

   try
      saveas(gcbf,filename);
   catch
      msg = 'ERROR: Cannot save file';
      set(findobj(gcf,'Tag','MessageLine'),'String',msg);
   end

   set(gcbf,'pointer',old_pointer);

   return;					% export_fig

