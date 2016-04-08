
clc

%--- using the Spectrally weighted Common Spatial Pattern (Spec-CSP) method ---
% this method automatically learns the spectral filters (within a pre-defined range), but it
% may run into a local optimum or over-fit spuriously correlated bands

% load the data set
traindata = io_loadset('/Users/Neda.Qusp/Qusp/Projects/ArtifactSubspaceRecusntruction/Data/WellBehavedWithSomeBlinks/imag.set','channels',1:32);

% define the approach
%myapproach = {'CSP' 'SignalProcessing',{'EpochExtraction',[0.5 3],'FIRFilter','on','Resampling','off'}};

myapproach = {'SpecCSP' 'SignalProcessing',{'EpochExtraction',[0.5 3],'FIRFilter','on','Resampling','off'}};

% learn a predictive model
% [trainloss,lastmodel,laststats] = bci_train('Data',traindata,'Approach',myapproach,'TargetMarkers',{'S  1', 'S  2'}, ...
%     'EvaluationScheme','off'); %#ok<>

[trainloss,lastmodel,laststats] = bci_train('Data',traindata,'Approach',myapproach,'TargetMarkers',{'S  1', 'S  2'});

% visualize results
bci_visualize(lastmodel)