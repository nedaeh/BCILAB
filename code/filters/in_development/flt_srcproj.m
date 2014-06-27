function signal = flt_srcproj(varargin)
% Project data through inverse weights.
% Inverse weights are returned by flt_sourceLocalize. 
%
% Tim Mullen, SCCN/INC/UCSD, Feb, 2014

if ~exp_beginfun('filter'), return; end

declare_properties('name','SourceProject', 'experimental',true,'follows',{'flt_fourier_bandpower'},'cannot_precede',{'flt_sourceLocalize'},'cannot_follow',{'set_makepos'},'independent_channels',false, 'independent_trials',false);

% extract some defaults
headmodel_default = 'resources:/headmodels/standard-Colin27-385ch.mat';
if ~onl_isonline
    hmObj = arg_extract(varargin,{'hmObj','HeadModelObject'},[],headmodel_default);
    hmObj = hlp_validateHeadModelObject(hmObj);
    if isempty(hmObj)
        ROINames = {};
    else
        % get the unique ROI names
        [tmp, idx] = unique(hmObj.atlas.label);
        ROINames = hmObj.atlas.label(sort(idx))';
    end
else
    ROINames = {};
end

arg_define(varargin, ...
    arg_norep({'signal','Signal'},[],[],'Signal structure. Must contain .data field with channel data'), ...
    arg({'hmObj','HeadModelObject'},headmodel_default,[],'Head model object generated by MOBILAB. See MOBILAB''s headModel class.'), ...
    arg_subtoggle({'colroi','CollapseRoi'},'on',...
    {...
        arg({'roiAtlasLabels','ROIAtlasLabels','AtlasROI','ROI'},ROINames,ROINames,'Regions of interest atlas labels. This is a cell array of strings corresponding to a subset of the labels stored in hmObj.atlas.label. If empty the all ROIs in the head model are selected.','type','logical'), ...
        arg({'rule','CollapseRule'},'sum',{'none','mean','sum','max','maxmag','median'},'Method for collapsing projected current over ROIs. Return the (mean, integral, max, etc) projected value within each ROI defined in signal.roiVerticesReduced. Result is stored in signal.srcproj. The signal.srcproj matrix will be reduced to [num_rois x num_samples].'), ...
    },'Collapse projected activity within rois'), ...
    arg({'preshaper','Preshaper'},@(x)x,[],'Reshaping function to apply to signal.data before projection.'), ...
    arg({'postshaper','Postshaper'},@(x)x,[],'Reshaping function to apply to signal.data after projection'), ...
    arg({'datafield','DataField'},'fourier_bandpower',[],'signal field to project') ...
    );

if ~isfield(signal,'srcweights_all')
    error('signal.srcweights_all not found. Please call flt_sourceLocalize() first');
end

if isempty(signal.srcweights_all)
    % no source weights
    signal.srcproj = [];
    return;
end
if isempty(datafield)
    datafield = 'data'; end

hmObj     = hlp_validateHeadModelObject(hmObj);
if isempty(hmObj)
    error('HeadModelObject was improperly defined. Exiting');
end
if isempty(colroi.roiAtlasLabels) %#ok
    if isempty(ROINames)
        % get the unique ROI names
        [tmp, idx] = unique(hmObj.atlas.label);
        ROINames = hmObj.atlas.label(sort(idx))';
    end
    colroi.roiAtlasLabels = ROINames;
end

% get the vertices for chosen ROIs
roiVertices = hlp_microcache('flt_srcproj',@GetRoiVertices,colroi.roiAtlasLabels,signal.rmIndices,hmObj);

% project data matrix through lead field matrix
if isempty(preshaper)
    preshaper  = @(x)x; end
if isempty(postshaper)
    postshaper = @(x)x; end

% project the data and collapse across ROIs
signal.srcproj = signal.srcweights_all*preshaper(signal.(datafield));
if colroi.arg_selection
    % TODO: get the ROI vertices
    signal.srcproj = hlp_colsrc(signal.srcproj,roiVertices,colroi.rule);
end
% apply postshaper
signal.srcproj = postshaper(signal.srcproj);

exp_endfun;


function roiVertices = GetRoiVertices(roiAtlasLabels,rmIndices,hmObj)
% We also store the indices of the vertices of each ROI (in the full
% source space) in a cell array. This allows us to obtain dipole
% centroids for each ROI

% get ROI indices into reduced source space. These are used for
% integration over current source density (CSD) within each ROI
nvert   = length(hmObj.atlas.color);
LFMcols = 1:nvert;
LFMcols(rmIndices) = [];
for k=1:length(roiAtlasLabels)
    roiVertices{k} = find(ismember(LFMcols,indices4Structure(hmObj,roiAtlasLabels{k})'));
end

