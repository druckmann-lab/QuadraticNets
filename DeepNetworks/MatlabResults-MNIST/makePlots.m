% close all;
% clear all;
% 
% %files = {'identityFullRank', 'identitySigns', 'gaussianMatrix', 'nonplanted', 'identityLowRank', 'lowRankMixedSigns', 'gaussianLowRank'};
% %files = {'gaussianMatrix', 'naturalGaussian', 'naturalIdentity'}
% %files = {'nonplantedNatural'}
% files = {'VectorDictionary', 'VectorDictionaryId'}
% 
% 
% for i = files

name = 'TwoLayerUniform'
% name = i{1};
% 
% in_file = strcat(name, '.mat');
load('TwoLayerUniform.mat')

hidden = [0:10];

threshold = residual_nn < 0.007;

% if(strcmp(name, 'nonplanted') | strcmp(name, 'nonplantedNatural'))
%     for i = 1:20
%         thresh = residual_lsq(i)+0.002
%         threshold(:, i) = (residual_nn(:, i) < thresh)
%     end
% else
%     threshold = (residual_nn < 0.011);
% 
% end

dims_thresh = size(threshold);
threshold_sum = vertcat(0, sum(threshold,2)/dims_thresh(2));

average = vertcat(1, mean(residual_nn, 2));

f1 = figure('visible','off')
plot(hidden, average, '-s', 'LineWidth', 2, 'MarkerSize', 12)
ax = gca;
ax.FontSize = 18; 
%ax.FontName = 'Myriad Pro'
ax.Box = 'Off'
ax.LineWidth = 2
ylim([0, 1])
ylabel('Average Normalized Error', 'fontsize', 18)
xlabel('Number of Hidden Units', 'fontsize', 18)
saveas(f1, strcat('Plots/average_', name), 'epsc')

f2 = figure('visible','off')
plot(hidden, threshold_sum, '-rs', 'LineWidth', 2, 'MarkerSize', 12)
ax = gca;
ax.FontSize = 18; 
%ax.FontName = 'Myriad Pro'
ax.Box = 'Off'
ax.LineWidth = 2
ylabel('Fraction Achieving Global Minimzer', 'fontsize', 18)
xlabel('Number of Hidden Units', 'fontsize', 18)
saveas(f2, strcat('Plots/threshold_', name), 'epsc')
