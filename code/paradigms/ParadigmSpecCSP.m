classdef ParadigmSpecCSP < ParadigmDataflowSimplified
    % Advanced paradigm for oscillatory processes via the Spectrally weighted CSP algorithm.
    %
    % The Spec-CSP paradigm [1] is a more advanced variant of CSP, developed for the Berlin
    % Brain-Computer Interface (BBCI); the primary focus was motor imagery BCI, but the algorithm was
    % designed from the outset to be applicable for a wider range of applications. The implementation
    % closely follows the TR [2].
    %
    % The paradigm is applicable to the majority of oscillatory processes, and is the most advanced
    % spatio-spectrally adaptive method that is currently provided in the toolbox. Whenever the exact
    % frequency and location of some (conjectured) oscillatory process is not known exactly, Spec-CSP
    % can be used, and typically gives better results than CSP with an appropriately unrestricted (e.g.,
    % broad-band) spectral filter. Several other methods exist to adapt the spectrum to a process of
    % interest, among others Common Spatio-Spectral Patterns [3], Common Sparse Spectral Spatial Pattern
    % [4], r^2-based heuristics [5], automated parameter search, and manual selection based on visual
    % inspection. Several of these methods have been shown to give approx. comparable results [2]. An
    % alternative and competitive method, especially when there complex interactions between frequency
    % bands and time periods are to be modeled is the Dual-Augmented Lagrange paradigm
    % (para_dal/para_dal_hf).
    %
    % The method iteratively optimizes spatial and spectral filters in alternation and extracts
    % log-variance features from the resulting (filtered) signal. These features are subsequently
    % processed by a (typically simple) machine learning component, by default LDA. Learning is
    % therefore significantly slower than CSP. An option is to impose custom prior knowledge on the
    % relevant data spectrum, for example by placing a focus on the alpha rhythm, without ruling out
    % other frequencies. Note that there are parameters which constrain the spectrum: one is the
    % frequency prior and the other is the spectral filter that is applied before running the alorithm;
    % both need to be adapted when the considered spectrum shall be extended (e.g. to high-gamma
    % oscillations). Other parameters which are frequently adapted are the time window of interest and
    % the learner component (e.g., logistic regression is a good alternative choice).
    %
    % Some application areas include detection of major brain rhythm modulations (e.g. theta, alpha,
    % beta, gamma), for example related to relaxation/stress, aspects of workload, emotion,
    % sensori-motor imagery, and in general cortical idle oscillations in various modalities.
    %
    % Example: Consider the goal of predicting the emotion felt by a person at a given time. A possible
    % calibration data set for this task would contain a sequence of blocks in each of which the subject
    % is one out of several possible emotions, indicated by events 'e1','e2','e3','e4' covering these
    % blocks at regular rate. The data might for example be induced via guided imagery [6].
    %
    %   calib = io_loadset('data sets/bruce/emotions.eeg')
    %   myapproach = {'SpecCSP' 'SignalProcessing',{'EpochExtraction',[-2.5 2.5]}};
    %   [loss,model,stats] = bci_train('Data',calib,'Approach',myapproach, 'TargetMarkers',{'e1','e2','e3','e4'});
    %
    %
    % References:
    %  [1] Tomioka, R., Dornhege, G., Aihara, K., and Mueller, K.-R. "An iterative algorithm for spatio-temporal filter optimization."
    %      In Proceedings of the 3rd International Brain-Computer Interface Workshop and Training Course 2006.
    %  [2] Ryota Tomioka, Guido Dornhege, Guido Nolte, Benjamin Blankertz, Kazuyuki Aihara, and Klaus-Robert Mueller
    %      "Spectrally Weighted Common Spatial Pattern Algorithm for Single Trial EEG Classification",
    %      Mathematical Engineering Technical Reports (METR-2006-40), July 2006.
    %  [3] Steven Lemm, Benjamin Blankertz, Gabriel Curio, and Klaus-Robert M�ller.
    %      "Spatio-spectral filters for improving classification of single trial EEG."
    %      IEEE Trans Biomed Eng, 52(9):1541-1548, 2005.
    %  [4] G. Dornhege, B. Blankertz, M. Krauledat, F. Losch, G. Curio, and K.-R. M�ller,
    %      "Combined optimization of spatial and temporal filters for improving brain-computer interfacing,"
    %      IEEE Transactions on Biomedical Engineering, vol. 53, no. 11, pp. 2274?2281, 2006.
    %  [5] Benjamin Blankertz, Ryota Tomioka, Steven Lemm, Motoaki Kawanabe, and Klaus-Robert Mueller.
    %      "Optimizing spatial filters for robust EEG single-trial analysis."
    %      IEEE Signal Process Mag, 25(1):41-56, January 2008
    %  [6] Onton J & Makeig S. "Broadband high-frequency EEG dynamics during emotion imagination."
    %      Frontiers in Human Neuroscience, 2009.
    %
    % Name:
    %   Spectrally Weighted CSP
    %
    %                           Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
    %                           2010-04-29
    
    methods
        
        function defaults = preprocessing_defaults(self)
            
            defaults = {'FIRFilter',{'Frequencies',[6 7 33 34],'Type','minimum-phase'}, 'EpochExtraction',[0.5 3.5], 'Resampling',100};
        end
        
        function model = feature_adapt(self,varargin)
            args = arg_define(varargin, ...
                arg_norep('signal'), ...
                arg({'patterns','PatternPairs'},3,uint32([1 1 64 1000]),'Number of CSP patterns (times two).','cat','Feature Extraction'), ...
                arg({'pp','ParameterP'},0,[-1 1],'Regularization parameter p''. Can be searched over -1:0.5:1.','cat','Feature Extraction','guru',true), ...
                arg({'qp','ParameterQ'},1,[0 4],'Regularization parameter q''. Can be searched over 0:0.5:4.','cat','Feature Extraction','guru',true), ...
                arg({'prior','SpectralPrior'},'@(f) f>=7 & f<=30',[],'Prior frequency weighting function.','cat','Feature Extraction', 'type','expression'), ...
                arg({'steps','MaxIterations'},3,uint32([1 3 10 50]),'Number of iterations. A step is spatial optimization, followed by spectral optimization.','cat','Feature Extraction'));
            
            
            [signal,n_of,pp,qp,prior,steps] = deal(args.signal,args.patterns,args.pp,args.qp,args.prior,args.steps);
            
            
            if signal.nbchan == 1
                error('Spec-CSP does intrinsically not support single-channel data (it is a spatial filter).'); end
            if signal.nbchan < args.patterns
                error('Spec-CSP prefers to work on at least as many channels as you request output patterns. Please reduce the number of pattern pairs.'); end
            
            
            % read a few parameters from the options (and re-parameterize the hyper-parameters p' and q' into p and q)
            p = pp+qp;
            q = qp;
            Fs = signal.srate;
            nfft = 2^(nextpow2(signal.pnts));           
            [freqs,findx] = getfgrid(Fs,nfft,[7 30]);            
            allfreqs = 0:Fs/nfft:Fs;
            allfreqs = allfreqs(1:nfft);
            
            SpectrumCalcAlgo = 'RegFFT'; %'RegFFT', 'Multitaper', 'Welch'
            
            % number of C=Channels, S=Samples and T=Trials #ok<NASGU>
            [C,S,dum] = size(signal.data); %#ok<NASGU>
            
            if strcmp(SpectrumCalcAlgo,'Multitaper')
                seq_length = S;
                time_halfbandwidth = 2.5;
                num_seq = 4;
                [dps_seq] = dpss(seq_length,time_halfbandwidth,num_seq);
            elseif strcmp(SpectrumCalcAlgo,'Welch')
                seglen = 100;
                overlap = 40;
                win = hamming(seglen);
                nwin = fix((S-overlap)/(seglen-overlap));
            end
            
            
            % preprocessing
            for c=1:2
                % compute the per-class epoched data X and its Fourier transform (along time), Xfft
                X{c} = exp_eval_optimized(set_picktrials(signal,'rank',c));
                [C,S,T] = size(X{c}.data);
                F{c} = zeros(length(freqs),C,C,T);
                
                
                if strcmp(SpectrumCalcAlgo,'RegFFT')
                    Xfft = fft(X{c}.data,nfft,2); %  Xfft->C,nfft,T
                    Xfft_crop = Xfft(:,findx,:);% Xfft_crop ->C,Freqs,T
                    F{c} = 2*real(bsxfun(@times, conj(permute(Xfft_crop,[1 4 2 3])), permute(Xfft_crop, [4 1 2 3]))); %F{c} -> C,C,freqs,T
                    
                    %                     spec_f_mine = (1/(S*2*pi))*(abs(Xfft_crop)).^2;
                    %                     spec_f_mine = squeeze(mean(spec_f_mine, 3));
                    %                     figure();
                    %                     plot(spec_f_mine');
                    %
                    %                     spec_f_temp = zeros(C,nfft,T);
                    %                     for cc=1:C
                    %
                    %                         spec_f_temp(cc,:,:) = periodogram(squeeze(X{c}.data(cc,:,:)),[],nfft,'twosided');
                    %                     end
                    %                     spec_f = spec_f_temp(:,findx,:);
                    %                     spec_f = squeeze(mean(spec_f, 3));
                    %                     figure();
                    %                     plot(spec_f');
                    %
                    
                    
                    
                elseif strcmp(SpectrumCalcAlgo,'Multitaper')
                    
                    
                    Xdata_win = bsxfun(@times,X{c}.data, reshape(dps_seq, 1, [], 1, size(dps_seq, 2))); % data_win -> C,S,T,num_seq
                    Xfft = fft(Xdata_win,nfft,2);   % Xfft->C,nfft,T,num_seq
                    Xfft_crop = Xfft(:,findx,:,:);  % Xfft_crop->C,freqs,T,num_seq
                    F{c} = 2* real(squeeze(sum(bsxfun(@times, conj(permute(Xfft_crop,[1,5,2,3,4])), permute(Xfft_crop, [5,1,2,3,4])), 5))./num_seq); % F{c}->C,C,freqs,T
                    
                    %save('/Users/Neda.Qusp/Qusp/Projects/SpatialFiltering/SourceCode/TestBuffer/mtm.mat','dps_seq','Xdata_win','Xfft','Xfft_crop')
                    %                     spec_mt_temp = zeros(C,nfft,T);
                    %                     for cc=1:C
                    %                         spec_mt_temp(cc,:,:) = pmtm(squeeze(X{c}.data(cc,:,:)),2.5,nfft,'twosided');
                    %                     end
                    %                     spec_mt = spec_mt_temp(:,findx,:);
                    %                     spec_mt = squeeze(mean(spec_mt, 3));
                    %                     figure();
                    %                     plot(spec_mt');
                    %
                    %                     spec_mt_mine = (1/(2*pi*num_seq))*sum((abs(Xfft_crop)).^2,4);
                    %                     spec_mt_mine = squeeze(mean(squeeze(spec_mt_mine), 3));
                    %                     figure();
                    %                     plot(spec_mt_mine');
                    
                    
                elseif strcmp(SpectrumCalcAlgo,'Welch')
                    
                    signal_idx = bsxfun(@plus,[1:seglen]',[0:nwin-1]*(seglen-overlap));
                    Xdata_seg = repmat(X{c}.data,1,1,1,nwin);  %Xdata_seg -> C,S,T,nwin
                    Xdata_seg2 = permute(Xdata_seg,[1 3 2  4]); %Xdata_seg2 ->C,T,S,nwin
                    Xdata_seg3 = reshape(Xdata_seg2(:,:,signal_idx),C,T,[],nwin); % Xdata_seg3 -> C,T,seglen,nwin
                    Xdata_win = bsxfun(@times,Xdata_seg3,reshape(win,1,1,seglen,1)); % Xdata_win -> C,T,seglen,nwin
                    Xfft = fft(Xdata_win,nfft,3); % Xfft -> C,T,nfft,nwin
                    Xfft_crop =Xfft(:,:,findx,:); % Xfft_crop->C,T,freqs,num_seq
                    F{c} = 2  * real(squeeze(sum(bsxfun(@times,conj(permute(Xfft_crop,[1,5,3,2,4])),permute(Xfft_crop,[5,1,3,2,4])),5))./nwin); % F{c}-> C,C,freqs,T
                    
                    %                     spec_w_temp = zeros(C,nfft,T);
                    %                     for cc=1:C
                    %                         spec_w_temp(cc,:,:) = pwelch(squeeze(X{c}.data(cc,:,:)),seglen,overlap,nfft,'twosided');
                    %                     end
                    %                     spec_w = spec_w_temp(:,findx,:);
                    %                     spec_w = squeeze(mean(spec_w, 3));
                    %                     figure();
                    %                     plot(spec_w');
                    %
                    %                     spec_w_mine = (1/(S*nwin))*sum((abs(Xfft_crop)).^2,4);
                    %                     spec_w_mine = squeeze(mean(permute(squeeze(spec_w_mine),[1,3,2]), 3));
                    %                     figure();
                    %                     plot(spec_w_mine');
                    %
                    
                end
                
                % compute the cross-spectrum V as an average over trials
                V{c} = mean(F{c},4);
                
            end
            
%             fpath = '/Users/Neda.Qusp/Qusp/Projects/SpatialFiltering/SourceCode/TestBuffer/';
%             if strcmp(SpectrumCalcAlgo,'RegFFT')
%                 fpath = strcat(fpath,'RegFFT/');
%             elseif strcmp(SpectrumCalcAlgo,'Multitaper')
%                 fpath = strcat(fpath,'Multitaper/');
%             elseif strcmp(SpectrumCalcAlgo,'Welch')
%                 fpath = strcat(fpath,'Welch/');
%             end
%                         
%             fname1 = strcat(fpath,'input.mat');
%             fname2 = strcat(fpath,'output1.mat');
%             X02save = X{1}.data;
%             X12save = X{2}.data;
%             save(fname1,'X02save','X12save');
%             F02save = F{1};
%             F12save = F{2};
%             V02save = V{1};
%             V12save = V{2};
%             save(fname2,'F02save','F12save','V02save','V12save');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            nF = length(freqs);
            
            % 1. initialize the filter set alpha and the number of filters J
            J = 1; alpha{J}(1:nF) = 1;
            % 2. for each step
            
            for step=1:steps
                % 3. for each set of spectral coefficients alpha{j} (j=1,...,J)
                for j=1:J
                    % 4. calculate sensor covariance matrices for each class from alpha{j}
                    for c = 1:2
                        Sigma{c} = zeros(C);
                        for b=1:nF
                            Sigma{c} = Sigma{c} + alpha{j}(b)*V{c}(:,:,b);
                        end
                    end
                    % 5. solve the generalized eigenvalue problem Eq. (2)
                    [VV,DD] = eig(Sigma{1},Sigma{1}+Sigma{2});
                    % and retain n_of top eigenvectors at both ends of the eigenvalue spectrum...
                    W{j} = {VV(:,1:n_of), VV(:,end-n_of+1:end)};
                    iVV = inv(VV)'; P{j} = {iVV(:,1:n_of), iVV(:,end-n_of+1:end)};
                    % as well as the top eigenvalue for each class
                    lambda(j,:) = [DD(1), DD(end)];
                end
                
                
                % 7. set W{c} from all W{j}{c} such that lambda(j,c) is minimal/maximal over j
                W = {W{argmin(lambda(:,1))}{1}, W{argmax(lambda(:,2))}{2}};
                P = {P{argmin(lambda(:,1))}{1}, P{argmax(lambda(:,2))}{2}};
                % 8. for each projection w in the concatenated [W{1},W{2}]...
                Wcat = [W{1} W{2}]; J = 2*n_of;
                Pcat = [P{1} P{2}];
                
                for j=1:J
                    w = Wcat(:,j);
                    % 9. calcualate (across trials within each class) mean and variance of the w-projected cross-spectrum components
                    for c=1:2
                        % part of Eq. (3)
                        s{c} = zeros(size(F{c},4),nF);
                        for k=1:nF
                            for t = 1:size(s{c},1)
                                s{c}(t,k) = w'*F{c}(:,:,k,t)*w;
                            end
                        end
                        mu_s{c} = mean(s{c});
                        var_s{c} = var(s{c});
                    end
                    % 10. update alpha{j} according to Eqs. (4) and (5)
                    for c=1:2
                        for k=1:nF
                            % Eq. (4)
                            alpha_opt{c}(k) = max(0, (mu_s{c}(k)-mu_s{3-c}(k)) / (var_s{1}(k) + var_s{2}(k)) );
                            
                            % Eq. (5), with prior from Eq. (6)
                            alpha_tmp{c}(k) = alpha_opt{c}(k).^q * ((mu_s{1}(k) + mu_s{2}(k))/2).^p;
                            
                        end
                    end
                    % ... as the maximum for both classes
                    alpha{j} = max(alpha_tmp{1},alpha_tmp{2});
                    % and normalize alpha{j} so that it sums to unity
                    alpha{j} = alpha{j} / sum(alpha{j});
                end
            end
            
            alphacat = zeros(nfft,J);
            alphacat(findx,:)= vertcat(alpha{:})';
            model = struct('W',{Wcat},'P',{Pcat},'alpha',{alphacat},'freqs',{allfreqs},'bands',{findx},'chanlocs',{signal.chanlocs});
            
            %fname3 = strcat(fpath,'output2.mat');
            %save(fname3,'Wcat','alphacat');
        end
        
        function features = feature_extract(self,signal,featuremodel)
            
            features = zeros(size(signal.data,3),size(featuremodel.W,2));
            nfft = 2^(nextpow2(signal.pnts));
            for t=1:size(signal.data,3)
                temp = signal.data(:,:,t)'*featuremodel.W;
                temp = fft(temp,nfft);
                temp = featuremodel.alpha.* temp;
                temp = 2*real(ifft(temp));
                temp = var(temp);
                temp = log(temp);
                features(t,:) = temp;
            end
        end
        
        function visualize_model(self,varargin) %#ok<*INUSD>
            args = arg_define([0 3],varargin, ...
                arg_norep({'myparent','Parent'},[],[],'Parent figure.'), ...
                arg_norep({'featuremodel','FeatureModel'},[],[],'Feature model. This is the part of the model that describes the feature extraction.'), ...
                arg_norep({'predictivemodel','PredictiveModel'},[],[],'Predictive model. This is the part of the model that describes the predictive mapping.'), ...
                arg({'patterns','PlotPatterns'},true,[],'Plot patterns instead of filters. Whether to plot spatial patterns (forward projections) rather than spatial filters.'), ...
                arg({'paper','PaperFigure'},false,[],'Use paper-style font sizes. Whether to generate a plot with font sizes etc. adjusted for paper.'), ...
                arg_nogui({'nosedir_override','NoseDirectionOverride'},'',{'','+X','+Y','-X','-Y'},'Override nose direction.'));
            arg_toworkspace(args);
            
            % no parent: create new figure
            if isempty(myparent)
                myparent = figure('Name','Common Spatial Patterns'); end
            % determine nose direction for EEGLAB graphics
            try
                nosedir = args.fmodel.signal.info.chaninfo.nosedir;
            catch
                disp_once('Nose direction for plotting not store in model; assuming +X');
                nosedir = '+X';
            end
            if ~isempty(nosedir_override)
                nosedir = nosedir_override; end
            % number of pairs, and index of pattern per subplot
            np = size(featuremodel.W,2)/2; idxp = [1:np np+(2*np:-1:np+1)]; idxf = [np+(1:np) 2*np+(2*np:-1:np+1)];
            % for each CSP pattern...
            for p=1:np*2
                subplot(4,np,idxp(p),'Parent',myparent);
                if args.patterns
                    plotdata = featuremodel.P(:,p);
                else
                    plotdata = featuremodel.W(:,p);
                end
                topoplot(plotdata,featuremodel.chanlocs,'nosedir',nosedir);
                subplot(4,np,idxf(p),'Parent',myparent);
                alpha = featuremodel.alpha(:,p);
                range = 1:max(find(alpha)); %#ok<MXFND>
                pl=plot(featuremodel.freqs(range),featuremodel.alpha(range,p));
                xlim([min(featuremodel.freqs(range)) max(featuremodel.freqs(range))]);
                l1 = xlabel('Frequency in Hz');
                l2 = ylabel('Weight');
                t=title(['Spec-CSP Pattern ' num2str(p)]);
                if args.paper
                    set([gca,t,l1,l2],'FontUnits','normalized');
                    set([gca,t,l1,l2],'FontSize',0.2);
                    set(pl,'LineWidth',2);
                end
            end
            try set(gcf,'Color',[1 1 1]); end
        end
        
        function layout = dialog_layout_defaults(self)
            layout = {'SignalProcessing.Resampling.SamplingRate', 'SignalProcessing.FIRFilter.Frequencies', ...
                'SignalProcessing.FIRFilter.Type', 'SignalProcessing.EpochExtraction', '', ...
                'Prediction.FeatureExtraction', '', ...
                'Prediction.MachineLearning.Learner'};
        end
        
        function tf = needs_voting(self)
            tf = true;
        end
        
    end
end
