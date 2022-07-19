%BIPOLAR returns an M-by-3 matrix containing a blue-red colormap, in
%	in which red stands for positive, blue stands for negative, 
%	and white stands for 0.
%
%  Usage: cmap = bipolar(M, lo, hi, contrast);  or  cmap = bipolar;
%
%  cmap:  output M-by-3 matrix for BIPOLAR colormap.
%  M:	  number of shades in the colormap. By default, it is the
%	  same length as the current colormap.
%  lo:	  the lowest value to represent.
%  hi:	  the highest value to represent.
%
%  Inspired from the LORETA PASCAL program:
%	http://www.unizh.ch/keyinst/NewLORETA
%
%  jimmy@rotman-baycrest.on.ca
%
%----------------------------------------------------------------
function cmap = bipolar(M, lo, hi, contrast)

   if ~exist('contrast','var')
      contrast = 128;
   end

   if ~exist('lo','var')
      lo = -1;
   end

   if ~exist('hi','var')
      hi = 1;
   end

   if ~exist('M','var')
      cmap = colormap;
      M = size(cmap,1);
   end

   steepness = 10 ^ (1 - (contrast-1)/127);
   pos_infs = 1e-99;
   neg_infs = -1e-99;

   doubleredc = [];
   doublebluec = [];

   if lo >= 0		% all positive

      if lo == 0
         lo = pos_infs;
      end

      for i=linspace(hi/M, hi, M)
         t = exp(log(i/hi)*steepness);
         doubleredc = [doubleredc; [(1-t)+t,(1-t)+0,(1-t)+0]];
      end

      cmap = doubleredc;

   elseif hi <= 0	% all negative

      if hi == 0
         hi = neg_infs;
      end

      for i=linspace(abs(lo)/M, abs(lo), M)
         t = exp(log(i/abs(lo))*steepness);
         doublebluec = [doublebluec; [(1-t)+0,(1-t)+0,(1-t)+t]];
      end

      cmap = flipud(doublebluec);

   else

      if hi > abs(lo)
         maxc = hi;
      else
         maxc = abs(lo);
      end

      for i=linspace(maxc/M, hi, round(M*hi/(hi-lo)))
         t = exp(log(i/maxc)*steepness);
         doubleredc = [doubleredc; [(1-t)+t,(1-t)+0,(1-t)+0]];
      end      

      for i=linspace(maxc/M, abs(lo), round(M*abs(lo)/(hi-lo)))
         t = exp(log(i/maxc)*steepness);
         doublebluec = [doublebluec; [(1-t)+0,(1-t)+0,(1-t)+t]];
      end

      cmap = [flipud(doublebluec); doubleredc];

   end

   return;					% bipolar

