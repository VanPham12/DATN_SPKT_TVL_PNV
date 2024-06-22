clear all; close all;
trainModel = 1;
rng(42)
if trainModel
    % Generate the training data
    [trainData,trainLabels] = hGenerateTrainingData(256);

    % Set the number of examples per mini-batch
    batchSize = 32;

    % Split real and imaginary grids into 2 image sets, then concatenate
    trainData = cat(4,trainData(:,:,1,:),trainData(:,:,2,:));
    trainLabels = cat(4,trainLabels(:,:,1,:),trainLabels(:,:,2,:));

    % Split into training and validation sets
    valData = trainData(:,:,:,1:batchSize);
    valLabels = trainLabels(:,:,:,1:batchSize);

    trainData = trainData(:,:,:,batchSize+1:end);
    trainLabels = trainLabels(:,:,:,batchSize+1:end); 

    % Validate roughly 5 times every epoch
    valFrequency = round(size(trainData,4)/batchSize/5); 
   
    % Define the CNN structure
    layers = [ ...
        imageInputLayer([612 14 1],'Normalization','none')
        convolution2dLayer(9,64,'Padding',4)
        reluLayer
        convolution2dLayer(5,64,'Padding',2,'NumChannels',64)
        reluLayer
        convolution2dLayer(5,64,'Padding',2,'NumChannels',64)
        reluLayer
        convolution2dLayer(5,32,'Padding',2,'NumChannels',64)
        reluLayer
        convolution2dLayer(5,1,'Padding',2,'NumChannels',32)
        regressionLayer
    ];


    % Set up a training policy
    options = trainingOptions('adam', ...
        'InitialLearnRate',3e-4, ...
        'MaxEpochs',5, ...
        'Shuffle','every-epoch', ...
        'Verbose',false, ...
        'Plots','training-progress', ...
        'MiniBatchSize',batchSize, ...
        'ValidationData',{valData, valLabels}, ...
        'ValidationFrequency',valFrequency, ...
        'ValidationPatience',5);

    % Train the network. The saved structure trainingInfo contains the
    % training progress for later inspection. This structure is useful for
    % comparing optimal convergence speeds of different optimization
    % methods.
    [channelEstimationCNN,trainingInfo] = trainNetwork(trainData, ...
        trainLabels,layers,options);

else
    % Load pretrained network if trainModel is set to false
    load('channelestimat_ha')
end

channelEstimationCNN.Layers

s1=zeros(1,10);
s2=zeros(1,10);
s3=zeros(1,10);
SNRdB = 1:10;
for SNRdB=1:10

%% 
% Load the predefined simulation parameters, including the PDSCH
% parameters and DM-RS configuration.
simParameters = hDeepLearningChanEstSimParameters();
carrier = simParameters.Carrier;
pdsch = simParameters.PDSCH;
%%
% Create a TDL channel model and set channel parameters. To compare
% different channel responses of the estimators, you can change these
% parameters later.

channel = nrTDLChannel;
channel.Seed = 0;
channel.DelayProfile = 'TDL-A';
channel.DelaySpread = 3e-7;
channel.MaximumDopplerShift = 50;

% This example supports only SISO configuration
channel.NumTransmitAntennas = 1;
channel.NumReceiveAntennas = 1;

waveformInfo = nrOFDMInfo(carrier);
channel.SampleRate = waveformInfo.SampleRate;


chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate))+chInfo.ChannelFilterDelay;               



% Generate DM-RS indices and symbols
dmrsSymbols = nrPDSCHDMRS(carrier,pdsch);
dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);

% Create resource grid
pdschGrid = nrResourceGrid(carrier);

% Map PDSCH DM-RS symbols to the grid
pdschGrid(dmrsIndices) = dmrsSymbols;

% OFDM-modulate associated resource elements
txWaveform = nrOFDMModulate(carrier,pdschGrid);

txWaveform = [txWaveform; zeros(maxChDelay,size(txWaveform,2))];
%% 
% Send data through the TDL channel model. 
[rxWaveform,pathGains,sampleTimes] = channel(txWaveform);

SNR = 10^(SNRdB/10); % Calculate linear SNR
N0 = 1/sqrt(2.0*simParameters.NRxAnts*double(waveformInfo.Nfft)*SNR);
noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
rxWaveform = rxWaveform + noise;
%% 
% Perform perfect synchronization. To find the strongest multipath
% component, use the information provided by the channel.
 
% Get path filters for perfect channel estimation
pathFilters = getPathFilters(channel); 
[offset,~] = nrPerfectTimingEstimate(pathGains,pathFilters);

rxWaveform = rxWaveform(1+offset:end, :);
%% 
% OFDM-demodulate the received data to recreate the resource grid.

rxGrid = nrOFDMDemodulate(carrier,rxWaveform);

% Pad the grid with zeros in case an incomplete slot has been demodulated
[K,L,R] = size(rxGrid);
if (L < carrier.SymbolsPerSlot)
    rxGrid = cat(2,rxGrid,zeros(K,carrier.SymbolsPerSlot-L,R));
end


estChannelGridPerfect = nrPerfectChannelEstimate(carrier,pathGains, ...
    pathFilters,offset,sampleTimes);
%% 
% To perform practical channel estimation, use the
% <docid:5g_ref#mw_function_nrChannelEstimate nrChannelEstimate> function.
[estChannelGrid,~] = nrChannelEstimate(carrier,rxGrid,dmrsIndices, ...
    dmrsSymbols,'CDMLengths',pdsch.DMRS.CDMLengths);

interpChannelGrid = hPreprocessInput(rxGrid,dmrsIndices,dmrsSymbols);

% Concatenate the real and imaginary grids along the batch dimension
nnInput = cat(4,real(interpChannelGrid),imag(interpChannelGrid));

% Use the neural network to estimate the channel
estChannelGridNN = predict(channelEstimationCNN,nnInput);

% Convert results to complex 
estChannelGridNN = complex(estChannelGridNN(:,:,:,1),estChannelGridNN(:,:,:,2));

%% 
% Calculate the mean squared error (MSE) of each estimation method.
neural_mse = mean(abs(estChannelGridPerfect(:) - estChannelGridNN(:)).^2);
s1(SNRdB)=neural_mse;
interp_mse = mean(abs(estChannelGridPerfect(:) - interpChannelGrid(:)).^2);
s2(SNRdB)=interp_mse;
practical_mse = mean(abs(estChannelGridPerfect(:) - estChannelGrid(:)).^2);
s3(SNRdB)=practical_mse;
%plotChEstimates(interpChannelGrid,estChannelGrid,estChannelGridNN,estChannelGridPerfect, interp_mse,practical_mse,neural_mse);
end
%%
% Plot the individual channel estimations and the actual channel
% realization obtained from the channel filter taps. Both the practical
% estimator and the neural network estimator outperform linear
% interpolation.
plotChEstimates(interpChannelGrid,estChannelGrid,estChannelGridNN,estChannelGridPerfect,...
    interp_mse,practical_mse,neural_mse)
figure;
semilogy(1:10, s1, 'Color', [1 0 0], 'Marker', 's', 'LineStyle', '-', 'LineWidth', 1.5); % màu đỏ cho đường đầu tiên
hold on;
semilogy(1:10, s2, 'Color', [0 0 1], 'Marker', 'o', 'LineStyle', '--', 'LineWidth', 1.5); % màu xanh cho đường thứ hai
semilogy(1:10, s3, 'Color', [1 0 1], 'LineStyle', '-', 'LineWidth', 1.5); % màu tím cho đường thứ ba
xlabel('SNR');
ylabel('mse');
legend('neural', 'interp', 'practical', 'location', 'best');
grid on;

%% Local Functions

function hest = hPreprocessInput(rxGrid,dmrsIndices,dmrsSymbols)
% Perform linear interpolation of the grid and input the result to the
% neural network This helper function extracts the DM-RS symbols from
% dmrsIndices locations in the received grid rxGrid and performs linear
% interpolation on the extracted pilots.

    % Obtain pilot symbol estimates
    dmrsRx = rxGrid(dmrsIndices);
    dmrsEsts = dmrsRx .* conj(dmrsSymbols);

    % Create empty grids to fill after linear interpolation
    [rxDMRSGrid, hest] = deal(zeros(size(rxGrid)));
    rxDMRSGrid(dmrsIndices) = dmrsSymbols;
    
    % Find the row and column coordinates for a given DMRS configuration
    [rows,cols] = find(rxDMRSGrid ~= 0);
    dmrsSubs = [rows,cols,ones(size(cols))];
    [l_hest,k_hest] = meshgrid(1:size(hest,2),1:size(hest,1));

    % Perform linear interpolation
    f = scatteredInterpolant(dmrsSubs(:,2),dmrsSubs(:,1),dmrsEsts);
    hest = f(l_hest,k_hest);

end

function [trainData,trainLabels] = hGenerateTrainingData(dataSize)
% Generate training data examples for channel estimation. Run dataSize
% number of iterations to create random channel configurations and pass an
% OFDM-modulated fixed resource grid with only the DM-RS symbols inserted.
% Perform perfect timing synchronization and OFDM demodulation, extracting
% the pilot symbols and performing linear interpolation at each iteration.
% Use perfect channel information to create the label data. The function
% returns 2 arrays - the training data and labels.

    fprintf('Starting data generation...\n')

    % List of possible channel profiles
    delayProfiles = {'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'};

    simParameters = hDeepLearningChanEstSimParameters();
    carrier = simParameters.Carrier;
    pdsch = simParameters.PDSCH;

    % Create the channel model object
    nTxAnts = simParameters.NTxAnts;
    nRxAnts = simParameters.NRxAnts;

    channel = nrTDLChannel; % TDL channel object
    channel.NumTransmitAntennas = nTxAnts;
    channel.NumReceiveAntennas = nRxAnts;

    % Use the value returned from <matlab:edit('nrOFDMInfo') nrOFDMInfo> to
    % set the channel model sample rate
    waveformInfo = nrOFDMInfo(carrier);
    channel.SampleRate = waveformInfo.SampleRate;

    % Get the maximum number of delayed samples by a channel multipath
    % component. This number is calculated from the channel path with the largest
    % delay and the implementation delay of the channel filter, and is required
    % to flush the channel filter to obtain the received signal.
    chInfo = info(channel);
    maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;

    % Return DM-RS indices and symbols
    dmrsSymbols = nrPDSCHDMRS(carrier,pdsch);
    dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);

    % Create resource grid
    grid = nrResourceGrid(carrier,nTxAnts);

    % PDSCH DM-RS precoding and mapping
    [~,dmrsAntIndices] = nrExtractResources(dmrsIndices,grid);
    grid(dmrsAntIndices) = dmrsSymbols;

    % OFDM modulation of associated resource elements
    txWaveform_original = nrOFDMModulate(carrier,grid);

    % Acquire linear interpolator coordinates for neural net preprocessing
    [rows,cols] = find(grid ~= 0);
    dmrsSubs = [rows, cols, ones(size(cols))];
    hest = zeros(size(grid));
    [l_hest,k_hest] = meshgrid(1:size(hest,2),1:size(hest,1));

    % Preallocate memory for the training data and labels
    numExamples = dataSize;
    [trainData, trainLabels] = deal(zeros([612 14 2 numExamples]));

    % Main loop for data generation, iterating over the number of examples
    % specified in the function call. Each iteration of the loop produces a
    % new channel realization with a random delay spread, doppler shift,
    % and delay profile. Every perturbed version of the transmitted
    % waveform with the DM-RS symbols is stored in trainData, and the
    % perfect channel realization in trainLabels.
    for i = 1:numExamples
        % Release the channel to change nontunable properties
        channel.release

        % Pick a random seed to create different channel realizations
        channel.Seed = randi([1001 2000]);

        % Pick a random delay profile, delay spread, and maximum doppler shift
        channel.DelayProfile = string(delayProfiles(randi([1 numel(delayProfiles)])));
        channel.DelaySpread = randi([1 300])*1e-9;
        channel.MaximumDopplerShift = randi([5 400]);

        % Send data through the channel model. Append zeros at the end of
        % the transmitted waveform to flush channel content. These zeros
        % take into account any delay introduced in the channel, such as
        % multipath delay and implementation delay. This value depends on
        % the sampling rate, delay profile, and delay spread
        txWaveform = [txWaveform_original; zeros(maxChDelay, size(txWaveform_original,2))];
        [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);

        % Add additive white Gaussian noise (AWGN) to the received time-domain
        % waveform. To take into account sampling rate, normalize the noise power.
        % The SNR is defined per RE for each receive antenna (3GPP TS 38.101-4).   
        SNRdB = randi([0 10]);  % Random SNR values between 0 and 10 dB
        SNR = 10^(SNRdB/10);    % Calculate linear SNR
        N0 = 1/sqrt(2.0*nRxAnts*double(waveformInfo.Nfft)*SNR);
        noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
        rxWaveform = rxWaveform + noise;

        % Perfect synchronization. Use information provided by the channel
        % to find the strongest multipath component
        pathFilters = getPathFilters(channel); % Get path filters for perfect channel estimation
        [offset,~] = nrPerfectTimingEstimate(pathGains,pathFilters);

        rxWaveform = rxWaveform(1+offset:end, :);

        % Perform OFDM demodulation on the received data to recreate the
        % resource grid, including padding in case practical
        % synchronization results in an incomplete slot being demodulated
        rxGrid = nrOFDMDemodulate(carrier,rxWaveform);
        [K,L,R] = size(rxGrid);
        if (L < carrier.SymbolsPerSlot)
            rxGrid = cat(2,rxGrid,zeros(K,carrier.SymbolsPerSlot-L,R));
        end

        % Perfect channel estimation, using the value of the path gains
        % provided by the channel. This channel estimate does not
        % include the effect of transmitter precoding
        estChannelGridPerfect = nrPerfectChannelEstimate(carrier,pathGains, ...
            pathFilters,offset,sampleTimes);

        % Linear interpolation
        dmrsRx = rxGrid(dmrsIndices);
        dmrsEsts = dmrsRx .* conj(dmrsSymbols);
        f = scatteredInterpolant(dmrsSubs(:,2),dmrsSubs(:,1),dmrsEsts);
        hest = f(l_hest,k_hest);

        % Split interpolated grid into real and imaginary components and
        % concatenate them along the third dimension, as well as for the
        % true channel response
        rx_grid = cat(3, real(hest), imag(hest));
        est_grid = cat(3, real(estChannelGridPerfect), ...
            imag(estChannelGridPerfect));

        % Add generated training example and label to the respective arrays
        trainData(:,:,:,i) = rx_grid;
        trainLabels(:,:,:,i) = est_grid;

        % Data generation tracker
        if mod(i,round(numExamples/25)) == 0
            fprintf('%3.2f%% complete\n',i/numExamples*100);
        end
    end
    fprintf('Data generation complete!\n')
end

function simParameters = hDeepLearningChanEstSimParameters()
% Set simulation parameters for Deep Learning Data Synthesis for 5G Channel Estimation example

    % Carrier configuration
    simParameters.Carrier = nrCarrierConfig;
    simParameters.Carrier.NSizeGrid = 51;            % Bandwidth in number of resource blocks (51 RBs at 30 kHz SCS for 20 MHz BW)
    simParameters.Carrier.SubcarrierSpacing = 30;    % 15, 30, 60, 120, 240 (kHz)
    simParameters.Carrier.CyclicPrefix = 'Normal';   % 'Normal' or 'Extended' (Extended CP is relevant for 60 kHz SCS only)
    simParameters.Carrier.NCellID = 2;               % Cell identity

    % Number of transmit and receive antennas
    simParameters.NTxAnts = 1;                      % Number of PDSCH transmission antennas
    simParameters.NRxAnts = 1;                      % Number of UE receive antennas

    % PDSCH and DM-RS configuration
    simParameters.PDSCH = nrPDSCHConfig;
    simParameters.PDSCH.PRBSet = 0:simParameters.Carrier.NSizeGrid-1; % PDSCH PRB allocation
    simParameters.PDSCH.SymbolAllocation = [0, simParameters.Carrier.SymbolsPerSlot];           % PDSCH symbol allocation in each slot
    simParameters.PDSCH.MappingType = 'A';     % PDSCH mapping type ('A'(slot-wise),'B'(non slot-wise))
    simParameters.PDSCH.NID = simParameters.Carrier.NCellID;
    simParameters.PDSCH.RNTI = 1;
    simParameters.PDSCH.VRBToPRBInterleaving = 0; % Disable interleaved resource mapping
    simParameters.PDSCH.NumLayers = 1;            % Number of PDSCH transmission layers
    simParameters.PDSCH.Modulation = '16QAM';                       % 'QPSK', '16QAM', '64QAM', '256QAM'

    % DM-RS configuration
    simParameters.PDSCH.DMRS.DMRSPortSet = 0:simParameters.PDSCH.NumLayers-1; % DM-RS ports to use for the layers
    simParameters.PDSCH.DMRS.DMRSTypeAPosition = 2;      % Mapping type A only. First DM-RS symbol position (2,3)
    simParameters.PDSCH.DMRS.DMRSLength = 1;             % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
    simParameters.PDSCH.DMRS.DMRSAdditionalPosition = 1; % Additional DM-RS symbol positions (max range 0...3)
    simParameters.PDSCH.DMRS.DMRSConfigurationType = 2;  % DM-RS configuration type (1,2)
    simParameters.PDSCH.DMRS.NumCDMGroupsWithoutData = 1;% Number of CDM groups without data
    simParameters.PDSCH.DMRS.NIDNSCID = 1;               % Scrambling identity (0...65535)
    simParameters.PDSCH.DMRS.NSCID = 0;                  % Scrambling initialization (0,1)
end

function plotChEstimates(interpChannelGrid,estChannelGrid,estChannelGridNN,estChannelGridPerfect,...
                         interp_mse,practical_mse,neural_mse)
% Plot the different channel estimates and display the measured MSE

    figure;
    cmax = max(abs([estChannelGrid(:); estChannelGridNN(:); estChannelGridPerfect(:)]));

    subplot(1,4,1)
    imagesc(abs(interpChannelGrid));
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    title({'Linear Interpolation', ['MSE: ', num2str(interp_mse)]});
    clim([0 cmax]);

    subplot(1,4,2)
    imagesc(abs(estChannelGrid));
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    title({'Practical Estimator', ['MSE: ', num2str(practical_mse)]});
    clim([0 cmax]);

    subplot(1,4,3)
    imagesc(abs(estChannelGridNN));
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    title({'Neural Network', ['MSE: ', num2str(neural_mse)]});
    clim([0 cmax]);

    subplot(1,4,4)
    imagesc(abs(estChannelGridPerfect));
    xlabel('OFDM Symbol');
    ylabel('Subcarrier');
    title({'Actual Channel'});
    clim([0 cmax]);

end