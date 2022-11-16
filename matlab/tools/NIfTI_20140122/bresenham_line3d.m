%  Generate X Y Z coordinates of a 3D Bresenham's line between
%  two given points.
%
%  A very useful application of this algorithm can be found in the
%  implementation of Fischer's Bresenham interpolation method in my
%  another program that can rotate three dimensional image volume
%  with an affine matrix:
%  http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=21080
%
%  Usage: [X Y Z] = bresenham_line3d(P1, P2, [precision]);
%
%  P1	- vector for Point1, where P1 = [x1 y1 z1]
%
%  P2	- vector for Point2, where P2 = [x2 y2 z2]
%
%  precision (optional) - Although according to Bresenham's line
%	algorithm, point coordinates x1 y1 z1 and x2 y2 z2 should
%	be integer numbers, this program extends its limit to all
%	real numbers. If any of them are floating numbers, you
%	should specify how many digits of decimal that you would
%	like to preserve. Be aware that the length of output X Y
%	Z coordinates will increase in 10 times for each decimal
%	digit that you want to preserve. By default, the precision
%	is 0, which means that they will be rounded to the nearest
%	integer.
%
%  X	- a set of x coordinates on Bresenham's line
%
%  Y	- a set of y coordinates on Bresenham's line
%
%  Z	- a set of z coordinates on Bresenham's line
%
%  Therefore, all points in XYZ set (i.e. P(i) = [X(i) Y(i) Z(i)])
%  will constitute the Bresenham's line between P1 and P1.
%
%  Example:
%	P1 = [12 37 6];     P2 = [46 3 35];
%	[X Y Z] = bresenham_line3d(P1, P2);
%	figure; plot3(X,Y,Z,'s','markerface','b');
%
%  This program is ported to MATLAB from:
%
%  B.Pendleton.  line3d - 3D Bresenham's (a 3D line drawing algorithm)
%  ftp://ftp.isc.org/pub/usenet/comp.sources.unix/volume26/line3d, 1992
%
%  Which is also referenced by:
%
%  Fischer, J., A. del Rio (2004).  A Fast Method for Applying Rigid
%  Transformations to Volume Data, WSCG2004 Conference.
%  http://wscg.zcu.cz/wscg2004/Papers_2004_Short/M19.pdf
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function [X,Y,Z] = bresenham_line3d(P1, P2, precision)

   if ~exist('precision','var') | isempty(precision) | round(precision) == 0
      precision = 0;
      P1 = round(P1);
      P2 = round(P2);
   else
      precision = round(precision);
      P1 = round(P1*(10^precision));
      P2 = round(P2*(10^precision));
   end

   d = max(abs(P2-P1)+1);
   X = zeros(1, d);
   Y = zeros(1, d);
   Z = zeros(1, d);

   x1 = P1(1);
   y1 = P1(2);
   z1 = P1(3);

   x2 = P2(1);
   y2 = P2(2);
   z2 = P2(3);

   dx = x2 - x1;
   dy = y2 - y1;
   dz = z2 - z1;

   ax = abs(dx)*2;
   ay = abs(dy)*2;
   az = abs(dz)*2;

   sx = sign(dx);
   sy = sign(dy);
   sz = sign(dz);

   x = x1;
   y = y1;
   z = z1;
   idx = 1;

   if(ax>=max(ay,az))			% x dominant
      yd = ay - ax/2;
      zd = az - ax/2;

      while(1)
         X(idx) = x;
         Y(idx) = y;
         Z(idx) = z;
         idx = idx + 1;

         if(x == x2)		% end
            break;
         end

         if(yd >= 0)		% move along y
            y = y + sy;
            yd = yd - ax;
         end

         if(zd >= 0)		% move along z
            z = z + sz;
            zd = zd - ax;
         end

         x  = x  + sx;		% move along x
         yd = yd + ay;
         zd = zd + az;
      end
   elseif(ay>=max(ax,az))		% y dominant
      xd = ax - ay/2;
      zd = az - ay/2;

      while(1)
         X(idx) = x;
         Y(idx) = y;
         Z(idx) = z;
         idx = idx + 1;

         if(y == y2)		% end
            break;
         end

         if(xd >= 0)		% move along x
            x = x + sx;
            xd = xd - ay;
         end

         if(zd >= 0)		% move along z
            z = z + sz;
            zd = zd - ay;
         end

         y  = y  + sy;		% move along y
         xd = xd + ax;
         zd = zd + az;
      end
   elseif(az>=max(ax,ay))		% z dominant
      xd = ax - az/2;
      yd = ay - az/2;

      while(1)
         X(idx) = x;
         Y(idx) = y;
         Z(idx) = z;
         idx = idx + 1;

         if(z == z2)		% end
            break;
         end

         if(xd >= 0)		% move along x
            x = x + sx;
            xd = xd - az;
         end

         if(yd >= 0)		% move along y
            y = y + sy;
            yd = yd - az;
         end

         z  = z  + sz;		% move along z
         xd = xd + ax;
         yd = yd + ay;
      end
   end

   if precision ~= 0
      X = X/(10^precision);
      Y = Y/(10^precision);
      Z = Z/(10^precision);
   end

   return;					% bresenham_line3d

