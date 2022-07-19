%  Imbed a zoom menu to any figure.
%
%  Usage: rri_zoom_menu(fig);
%

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
%--------------------------------------------------------------------
function menu_hdl = rri_zoom_menu(fig)

   if isnumeric(fig)
      menu_hdl = uimenu('Parent',fig, ...
   	   'Label','Zoom on', ...
	   'Userdata', 1, ...
           'Callback','rri_zoom_menu(''zoom'');');

      return;
   end

   zoom_on_state = get(gcbo,'Userdata');

   if (zoom_on_state == 1)
      zoom on;
      set(gcbo,'Userdata',0,'Label','Zoom off');
      set(gcbf,'pointer','crosshair');
   else
      zoom off;
      set(gcbo,'Userdata',1,'Label','Zoom on');
      set(gcbf,'pointer','arrow');
   end

   return					% rri_zoom_menu

