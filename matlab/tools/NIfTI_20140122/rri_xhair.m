%   rri_xhair: create a pair of full_cross_hair at point [x y] in
%              axes h_ax, and return xhair struct
%
%   Usage: xhair = rri_xhair([x y], xhair, h_ax);
%
%   If omit xhair, rri_xhair will create a pair of xhair; otherwise,
%   rri_xhair will update the xhair. If omit h_ax, current axes will
%   be used.
%

%   24-nov-2003 jimmy (jimmy@rotman-baycrest.on.ca)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function xhair = rri_xhair(varargin)

   if nargin == 0
      error('Please enter a point position as first argument');
      return;
   end

   if nargin > 0
      p = varargin{1};

      if ~isnumeric(p) | length(p) ~= 2
         error('Invalid point position');
         return;
      else
         xhair = [];
      end
   end

   if nargin > 1
      xhair = varargin{2};

      if ~isempty(xhair)
         if ~isstruct(xhair)
            error('Invalid xhair struct');
            return;
         elseif ~isfield(xhair,'lx') | ~isfield(xhair,'ly')
            error('Invalid xhair struct');
            return;
         elseif ~ishandle(xhair.lx) | ~ishandle(xhair.ly)
            error('Invalid xhair struct');
            return;
         end

         lx = xhair.lx;
         ly = xhair.ly;
      else
         lx = [];
         ly = [];
      end
   end

   if nargin > 2
      h_ax = varargin{3};

      if ~ishandle(h_ax)
         error('Invalid axes handle');
         return;
      elseif ~strcmp(lower(get(h_ax,'type')), 'axes')
         error('Invalid axes handle');
         return;
      end
   else
      h_ax = gca;
   end

   x_range = get(h_ax,'xlim');
   y_range = get(h_ax,'ylim');

   if ~isempty(xhair)
      set(lx, 'ydata', [p(2) p(2)]);
      set(ly, 'xdata', [p(1) p(1)]);
      set(h_ax, 'selected', 'on');
      set(h_ax, 'selected', 'off');
   else
      figure(get(h_ax,'parent'));
      axes(h_ax);

      xhair.lx = line('xdata', x_range, 'ydata', [p(2) p(2)], ...
	'zdata', [11 11], 'color', [1 0 0], 'hittest', 'off');
      xhair.ly = line('xdata', [p(1) p(1)], 'ydata', y_range, ...
	'zdata', [11 11], 'color', [1 0 0], 'hittest', 'off');
   end

   set(h_ax,'xlim',x_range);
   set(h_ax,'ylim',y_range);

   return;

