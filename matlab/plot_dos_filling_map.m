%PLOT_DOS_FILLING_MAP  DOS at E_F vs gate filling (X) and field (Y).
%
%   Reads a pre-computed DOS map (scripts/dos_map.py) and plots the DOS at
%   the Fermi level against gate voltage and magnetic field.
%
%   Two different "fillings" live in these files and they are not the same
%   thing:
%
%     filling_moire  states per moire cell. Linear in gate voltage, and
%                    independent of B. Runs -4 to +4 over the full dU range.
%                    This is the natural x-axis for a gate sweep, and the
%                    data grid is rectangular in it, so no resampling.
%     nu             Landau filling factor n*h/(e*B). Depends on B, so it is
%                    a 2-D array here. Lines of constant nu are straight
%                    lines through the origin of the (filling, B) plane --
%                    overlaid below, since that is how a fan is read.
%
%   Unlike the level-based scripts, the broadening here was applied on the
%   Python side and is baked into the data.  It is read from the file and
%   shown in the title; changing it means re-running scripts/dos_map.py.

clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(here);

% ======================================================================
%                              SETTINGS
% ======================================================================

MODEL_TAG = getenv('ABC_MODEL');
if isempty(MODEL_TAG), MODEL_TAG = 'six_band_minimal'; end
GRID_TAG  = getenv('ABC_GRID');
if isempty(GRID_TAG),  GRID_TAG  = '100x241'; end

datafile = fullfile(here, '..', 'data', ...
                    ['dos_map_' MODEL_TAG '_' GRID_TAG '.mat']);

% Colour scaling. The raw DOS varies only weakly across most of the map --
% the oscillations ride on a large smooth background -- so keying the scale
% to the global peak washes the fan out completely. Two fixes, both on by
% default:
%   CLIM_PCT      clip to a percentile range instead of to the peak
%   NORMALISE_ROWS divide each field by its own median, so the fan is shown
%                  as a fractional modulation. Set false for absolute DOS.
NORMALISE_ROWS = true;
CLIM_PCT  = [2 98];
LOG_SCALE = false;
NU_LINES  = [-24 -16 -12 -8 -4 4 8 12 16 24];   % constant-nu guides
SHOW_EXTRA = true;     % second figure: Fermi level and top-layer fraction
% ======================================================================

S = load(datafile);
f  = S.filling_moire(:);        % states per moire cell
B  = S.B_T(:);
D  = S.dos_cm2_eV;              % (nB x nf)
Draw = D;
if NORMALISE_ROWS
    D = D ./ median(D, 2);      % fractional modulation at each field
end

fprintf('%s\n', S.model);
fprintf('B = %.2f to %.2f T (%d points)\n', B(1), B(end), numel(B));
fprintf('moire filling = %.2f to %.2f (%d points), dU = %.0f to %.0f meV\n', ...
        f(1), f(end), numel(f), S.dU_meV(1), S.dU_meV(end));
fprintf('Landau nu spans %.1f at %.0f T to %.1f at %.0f T\n', ...
        max(S.nu(1, :)), B(1), max(S.nu(end, :)), B(end));
fprintf('broadening (fixed in the data): %s, gamma0 = %.3f meV, %s\n', ...
        S.broadening.lineshape, S.broadening.gamma0_eV*1e3, ...
        S.broadening.scaling);
fprintf('charge neutrality at %.3f to %.3f meV\n', ...
        min(S.cnp_meV), max(S.cnp_meV));

% ---------------------------------------------------------------- plot
figure('Color', 'w', 'Position', [80 80 900 620]);
imagesc(f, B, D); set(gca, 'YDir', 'normal');
if LOG_SCALE
    set(gca, 'ColorScale', 'log');
end
clim(pct(D, CLIM_PCT));
colormap(parula);
cb = colorbar;
if NORMALISE_ROWS
    cb.Label.String = 'DOS at E_F / median at that field';
else
    cb.Label.String = 'DOS at E_F  (cm^{-2} eV^{-1})';
end
hold on;

% constant Landau filling: nu = n h/(eB) and filling_moire = n*A, so
% filling_moire = nu * (e B / h) * A -- straight lines through the origin
fprintf('raw DOS spans %.3g to %.3g cm^-2 eV^-1\n', ...
        min(Draw(:)), max(Draw(:)));
A_moire = S.filling_moire(end) / (S.n_m2(end) + eps);
for v = NU_LINES
    y = B;
    x = v * (S.nphi_m2(:)) * A_moire;
    plot(x, y, 'w-', 'LineWidth', 0.6, 'Color', [1 1 1 0.45]);
    k = find(abs(x) <= max(abs(f)), 1, 'last');
    if ~isempty(k)
        text(x(k), y(k), sprintf('%+d', v), 'Color', 'w', 'FontSize', 6.5, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end
xline(0, 'w-', 'Alpha', 0.4);
xlim([min(f) max(f)]); ylim([min(B) max(B)]);
xlabel('gate filling  (states per moire cell)');
ylabel('magnetic field  B  (T)');
title([strrep(S.model, '_', '\_') ...
       ':  DOS at E_F.   \gamma = ' ...
       num2str(S.broadening.gamma0_eV*1e3, '%.2f') ...
       ' meV\times\surd(B/1T), ' S.broadening.lineshape ...
       '.   white lines: constant Landau \nu']);
box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

if SHOW_EXTRA
    figure('Color', 'w', 'Position', [120 120 1180 440]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    imagesc(f, B, S.mu_meV); set(gca, 'YDir', 'normal');
    colorbar; colormap(gca, parula);
    xlabel('gate filling'); ylabel('B  (T)');
    title('Fermi level  (meV)');
    box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

    nexttile;
    imagesc(f, B, S.top_fraction); set(gca, 'YDir', 'normal'); clim([0 1]);
    colorbar; colormap(gca, parula);
    xlabel('gate filling');
    title('fraction of DOS(E_F) on the top layer');
    box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

    nexttile;
    imagesc(f, B, S.nu); set(gca, 'YDir', 'normal');
    colorbar; colormap(gca, parula);
    xlabel('gate filling');
    title('Landau filling \nu  (depends on B)');
    box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

    title(tl, 'Context: the two fillings are different quantities');
end


% ----------------------------------------------------------------------
function lim = pct(X, p)
%PCT  Percentile limits without the Statistics Toolbox.
v = sort(X(isfinite(X)));
if isempty(v), lim = [0 1]; return; end
idx = max(1, min(numel(v), round(p/100 * numel(v))));
lim = [v(idx(1)), v(idx(2))];
if lim(2) <= lim(1), lim = [v(1), v(end) + eps]; end
end
