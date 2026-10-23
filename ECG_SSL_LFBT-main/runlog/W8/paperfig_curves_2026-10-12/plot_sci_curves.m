function plot_sci_curves()
% Reproducible publication plots for Figures 6, 7, 9, and 10.
% Outputs vector PDF plus 600-dpi TIFF. MATLAB R2021a or newer is recommended.

dataDir = fileparts(mfilename('fullpath'));
outDir = fullfile(dataDir, 'sci_exports');
if ~exist(outDir, 'dir'), mkdir(outDir); end

set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 8);
set(groot, 'defaultAxesLineWidth', 0.75);
set(groot, 'defaultLineLineWidth', 1.3);

blue = [0.000 0.447 0.698];
orange = [0.902 0.624 0.000];
green = [0.000 0.620 0.451];
vermillion = [0.835 0.369 0.000];
purple = [0.800 0.475 0.655];
gray = [0.400 0.400 0.400];

%% Figure 6: pretraining loss trajectories
T6 = readtable(fullfile(dataDir, 'fig6_pretrain_curves.csv'), 'TextType', 'string');
epochs6 = (0:99)';
branches = ["c1_lfbt", "c2_trc", "beat_rr", "noise_iso"];
names = ["C1 baseline", "C2 + TRC", "Beat-RR", "Noise"];
cols = [gray; green; purple; orange];

f = new_figure(); hold on;
for k = 1:numel(branches)
    y = nan(size(epochs6));
    for i = 1:numel(epochs6)
        idx = T6.branch == branches(k) & T6.seed == 0 & T6.epoch == epochs6(i);
        y(i) = log10(mean(T6.loss(idx), 'omitnan'));
    end
    plot(epochs6, y, 'Color', cols(k,:), 'DisplayName', names(k));
end
xlabel('Epoch'); ylabel('log_{10}(total loss)');
legend('Location', 'southoutside', 'NumColumns', 4, 'Box', 'off');
box off; grid on; set(gca, 'GridColor', [0.88 0.88 0.88], 'GridAlpha', 1);
export_pair(f, outDir, 'Fig6_pretraining_loss');

%% Figure 7: downstream validation loss and final test metrics
T7 = readtable(fullfile(dataDir, 'fig7_downstream_curves.csv'), 'TextType', 'string');
T7.val_loss_num = str2double(regexprep(string(T7.val_loss), '\.$', ''));
M7 = readtable(fullfile(dataDir, 'fig7_final_metrics.csv'), 'TextType', 'string');
epochs7 = (1:100)';
phases = ["lp", "ft10", "lpft"];
phaseNames = ["LP", "FT10", "LP-FT"];
phaseCols = [blue; vermillion; orange];

f = new_figure();
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; hold on;
for k = 1:numel(phases)
    y = nan(size(epochs7));
    for i = 1:numel(epochs7)
        idx = T7.phase == phases(k) & T7.epoch == epochs7(i);
        y(i) = mean(T7.val_loss_num(idx), 'omitnan');
    end
    plot(epochs7, y, 'Color', phaseCols(k,:), 'DisplayName', phaseNames(k));
end
xlabel('Epoch'); ylabel('Validation loss');
legend('Location', 'northeast', 'Box', 'off'); box off; grid on;

nexttile;
mu = zeros(3,2); sd = zeros(3,2);
for k = 1:numel(phases)
    idx = M7.phase == phases(k);
    mu(k,:) = [mean(M7.auroc(idx)), mean(M7.auprc(idx))];
    sd(k,:) = [std(M7.auroc(idx),0), std(M7.auprc(idx),0)];
end
b = bar(mu, 'grouped');
b(1).FaceColor = green; b(2).FaceColor = purple;
hold on;
for j = 1:2
    x = b(j).XEndPoints;
    errorbar(x, mu(:,j), sd(:,j), 'k', 'LineStyle', 'none', 'CapSize', 5);
    for i = 1:3
        text(x(i), mu(i,j) + sd(i,j) + 0.006, sprintf('%.3f', mu(i,j)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8);
    end
end
set(gca, 'XTickLabel', phaseNames); ylabel('Test score'); ylim([0.65 1.00]);
legend({'AUROC','AUPRC'}, 'Location', 'southoutside', 'Box', 'off'); box off; grid on;
export_pair(f, outDir, 'Fig7_downstream_loss_metrics');

%% Figure 9: label fraction with cross-seed variability
x = [10 20 40];
seedValues = [0.8627 0.8795 0.8856; 0.8611 0.8832 0.8892; 0.8554 0.8762 0.8828];
mu9 = mean(seedValues,1); sd9 = std(seedValues,0,1);
f = new_figure(); hold on;
offsets = [-1.3 0 1.3]; seedCols = [blue; gray; orange]; marks = {'o','s','^'};
seedIds = [0 2 4];
labelDy = [0.0026 0.0018 -0.0017];
for s = 1:3
    scatter(x + offsets(s), seedValues(s,:), 28, seedCols(s,:), marks{s}, 'filled', ...
        'DisplayName', sprintf('seed %d', seedIds(s)));
    for i = 1:3
        text(x(i)+offsets(s), seedValues(s,i)+labelDy(s), sprintf('%.4f', seedValues(s,i)), ...
            'VerticalAlignment', 'middle', 'HorizontalAlignment', 'center', ...
            'FontWeight', 'bold', 'FontSize', 8, 'Color', 'k');
    end
end
errorbar(x, mu9, sd9, '-d', 'Color', green, 'MarkerFaceColor', green, ...
    'MarkerSize', 6, 'CapSize', 8, 'DisplayName', 'mean ± SD');
for i = 1:3
    text(x(i), mu9(i)-0.0012, sprintf('%.4f', mu9(i)), 'VerticalAlignment', 'middle', ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8, 'Color', 'k');
end
xlabel('Labeled training data (%)'); ylabel('Macro-AUROC');
xlim([7 43]); ylim([0.85 0.895]); box off; grid on;
legend('Location', 'southoutside', 'NumColumns', 4, 'Box', 'off');
export_pair(f, outDir, 'Fig9_label_fraction');

%% Figure 10: matched TRC controls
f = new_figure();
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot_labeled([0.7589 0.7583 0.7627], blue, 'o', 'C1', 0.0005); hold on;
plot_labeled([0.7555 0.7523 0.7585], gray, 's', 'Affine', -0.0008);
plot_labeled([0.7659 0.7642 0.7747], green, 'd', 'C2 = C1 + TRC', 0.0006);
set(gca, 'XTick', 1:3, 'XTickLabel', {'seed 0','seed 2','seed 4'});
ylabel('Macro-AUPRC'); xlim([0.78 3.22]); ylim([0.745 0.780]); box off; grid on;
legend('Location', 'southoutside', 'Box', 'off');

nexttile;
plot_labeled([0.7775 0.7708 0.7712], blue, 'o', 'B0', -0.0007); hold on;
plot_labeled([0.7835 0.7778 0.7835], green, 'd', 'B0-TRC', 0.0006);
set(gca, 'XTick', 1:3, 'XTickLabel', {'seed 0','seed 2','seed 4'});
ylabel('Macro-AUPRC'); xlim([0.78 3.22]); ylim([0.765 0.790]); box off; grid on;
legend('Location', 'southoutside', 'Box', 'off');
export_pair(f, outDir, 'Fig10_matched_TRC_controls');

fprintf('SCI curve files written to: %s\n', outDir);
end

function f = new_figure()
f = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 19 10.7]);
end

function plot_labeled(y, color, marker, label, dy)
x = 1:numel(y);
plot(x, y, ['-' marker], 'Color', color, 'MarkerFaceColor', color, ...
    'MarkerSize', 6, 'DisplayName', label); hold on;
for i = 1:numel(y)
    text(x(i), y(i)+dy, sprintf('%.4f', y(i)), 'VerticalAlignment', 'middle', ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8, 'Color', 'k');
end
end

function export_pair(f, outDir, stem)
stem = string(stem);
exportgraphics(f, fullfile(outDir, stem + ".pdf"), 'ContentType', 'vector');
exportgraphics(f, fullfile(outDir, stem + ".tif"), 'Resolution', 600, 'BackgroundColor', 'white');
close(f);
end
