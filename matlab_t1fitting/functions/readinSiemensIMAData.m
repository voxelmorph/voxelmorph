function [WIPOutputArray, sqeuenceID] = readinSiemensIMAData(file, legacyOption)

if nargin<2
  legacyOption = 0;
end

dicomIMA = fileread(file);

delimBegin = strfind(dicomIMA,'### ASCCONV BEGIN ');
delimEnd = strfind(dicomIMA,'### ASCCONV END ###');

delimiters = strfind(dicomIMA,'###');
delimiters = delimiters( delimiters > delimBegin(1) );

dicomIMA = dicomIMA(delimiters(1)+4:delimEnd(1)-2);
ASCCFields = strtrim(textscan(dicomIMA,'%q%q','Delimiter','=\n'));

%% Read AlFree
% alFree = zeros( getCellValueByName(ASCCFields, 'sWipMemBlock.alFree.__attribute__.size', 'numeric') , 1);
alFree = zeros( 64 , 1);

[vals,names] = getCellValueByName(ASCCFields, 'sWipMemBlock.alFree', 'numeric');
rmInd = strcmp(names,'sWipMemBlock.alFree.__attribute__.size');
vals(rmInd) = [];
names(rmInd) = [];
names = cellfun( @lower, names, 'UniformOutput', 0);
pos = textscan(sprintf('%s\n',names{:}),lower('sWipMemBlock.alFree[%d]'));

alFree( pos{1} + 1 ) = vals;


%% Read AdFree
% adFree = zeros( getCellValueByName(ASCCFields, 'sWipMemBlock.adFree.__attribute__.size','numeric') , 1);
% 
% [vals,names] = getCellValueByName(ASCCFields, 'sWipMemBlock.adFree', 'numeric');
% rmInd = strcmp(names,'sWipMemBlock.adFree.__attribute__.size');
% vals(rmInd) = [];
% names(rmInd) = [];
% if ~isempty( vals )
%     pos = textscan(sprintf('%s\n',names{:}),'sWipMemBlock.adFree[%d]');
% 
%     adFree( pos{1} + 1 ) = vals;
% end


%% Only Output the CV_EXPORT array
if (legacyOption)
    WIPOutputArray = alFree(48:64);
else    
    WIPOutputArray = alFree(33:64);
end


%% Get other Sequence values
sqeuenceID = getCellValueByName(ASCCFields, 'lSequenceID','numeric');




end
function [val, names] = getCellValueByName(ASCCFields,fieldName,varargin)

    if ( (nargin >= 3) && strncmpi(varargin{1},'numeric',7) )
        convertFun = @str2double;
    else
        convertFun = @(x) x;
    end

    findInd = ~cellfun(@isempty,strfind(lower(ASCCFields{1}),lower(fieldName)));
    val = convertFun(ASCCFields{2}( findInd ));
    names = ASCCFields{1}( findInd );
end