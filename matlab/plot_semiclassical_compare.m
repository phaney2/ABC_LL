%PLOT_SEMICLASSICAL_COMPARE  Exact vs semiclassical Landau levels, all models.
%
%   One panel per model at a single interlayer potential: exact levels in
%   grey, semiclassical SBM (orbit area + Berry phase + orbital moment)
%   drawn over them.
%
%   Both valleys are merged, because levels_<model>.mat stores them merged.
%
%   Generate the data with, for each model:
%       python3 scripts/export_semiclassical.py six_band

clear; close all;

% ======================================================================
%                              SETTINGS
% ======================================================================

DU_PLOT = 0.10;      % interlayer potential, eV.  Must be nonzero: at dU = 0
                     % the band touching is gapless and the Berry curvature
                     % is a delta function the k-grid cannot see, so SBM
                     % silently collapses onto bare Onsager.
B_LIM   = [2 12];    % field range, T
E_LIM   = [-160 160];% energy range, meV

MODELS  = {'6band', '6band_minimal', '2band'};
TERM    = 'SBM';     % 'S' | 'SB' | 'SBM'

% ======================================================================

tags = struct('x6band', 'six_band', 'x6band_minimal', 'six_band_minimal', ...
              'x2band', 'two_band_minimal');
here = fileparts(mfilename('fullpath'));
CSEMI = [0.00 0.45 0.74];

if DU_PLOT == 0
    warning('abc:gapless', ['dU = 0 is gapless, so the Berry phase is ' ...
        'absent and SBM reduces to S.  Pick a nonzero dU.']);
end

figure('Color', 'w', 'Position', [60 80 1320 450]);
tl = tiledlayout(1, numel(MODELS), 'TileSpacing', 'compact', ...
                 'Padding', 'compact');

for im = 1:numel(MODELS)
    tag = tags.(['x' MODELS{im}]);
    fs = fullfile(here, '..', 'data', ['semiclassical_' tag '.mat']);
    fe = fullfile(here, '..', 'data', ['levels_' tag '.mat']);
    if ~isfile(fs)
        error('abc:nodata', ['%s not found.\nGenerate it with:\n' ...
            '    python3 scripts/export_semiclassical.py %s'], fs, tag);
    end
    if ~isfile(fe)
        error('abc:nodata', ['%s not found.\nGenerate it with:\n' ...
            '    python3 scripts/export_for_matlab.py %s'], fe, tag);
    end
    S = load(fs);  X = load(fe);

    [~, iu] = min(abs(S.dU_eV(:) - DU_PLOT));      % semiclassical dU index
    [~, ju] = min(abs(X.dU_eV(:) - DU_PLOT));      % exact dU index
    B = S.Blist(:);

    if im == 1
        fprintf('dU = %.0f meV (semiclassical), %.0f meV (exact),  %s\n', ...
                S.dU_eV(iu)*1e3, X.dU_eV(ju)*1e3, TERM);
    end

    nexttile(im); hold on;
    for a = 1:numel(X.B_T)
        if X.B_T(a) < B_LIM(1) || X.B_T(a) > B_LIM(2), continue; end
        n = X.n_levels(a, ju);
        if n == 0, continue; end
        e = reshape(X.energy_eV(a, ju, 1:n), 1, []) * 1e3;
        e = e(e >= E_LIM(1) & e <= E_LIM(2));
        hE = plot(X.B_T(a)*ones(size(e)), e, '.', 'Color', [0.75 0.75 0.75], ...
                  'MarkerSize', 4);
    end
    L = semi_levels(S, iu, TERM);                  % (nB, nLevels)
    for n = 1:size(L, 2)
        m = isfinite(L(:, n)) & L(:, n) >= E_LIM(1) & L(:, n) <= E_LIM(2) ...
            & B >= B_LIM(1) & B <= B_LIM(2);
        if any(m)
            hS = plot(B(m), L(m, n), '-', 'Color', CSEMI, 'LineWidth', 0.9);
        end
    end

    xlim(B_LIM); ylim(E_LIM); box on; grid on;
    set(gca, 'Layer', 'top', 'TickDir', 'out');
    title(strrep(MODELS{im}, '_', '\_'));
    xlabel('magnetic field  B  (T)');
    if im == 1, ylabel('energy  E  (meV)'); end
    if im == numel(MODELS) && exist('hS', 'var')
        legend([hE hS], {'exact', 'semiclassical'}, 'Location', 'northwest', ...
               'Box', 'off');
    end
end

title(tl, sprintf(['exact vs semiclassical Landau levels,  \\DeltaU = ' ...
    '%.0f meV,  %s'], DU_PLOT*1e3, terms_label(TERM)));

% ======================================================================
function L = semi_levels(S, iu, term)
%SEMI_LEVELS  All semiclassical levels at one dU, as (nB, nLevels).
%   Merges both valleys, both bands, every pocket and every Lifshitz
%   segment -- levels_<model>.mat stores the valleys merged, so the
%   comparison has to merge them too.
names = fieldnames(S);
pat = ['^LL_(K|Kp)_b\d+_p\d+_' term '(_seg\d+)?$'];
keys = names(~cellfun(@isempty, regexp(names, pat, 'once')));
L = [];
for k = 1:numel(keys)
    L = [L, squeeze(S.(keys{k})(iu, :, :))];   %#ok<AGROW>  (nB, nmax+1)
end
end

function s = terms_label(t)
switch upper(t)
    case 'S',   s = 'orbit area only';
    case 'SB',  s = 'area + Berry phase';
    case 'SBM', s = 'area + Berry phase + orbital moment';
    otherwise,  error('abc:terms', 'TERM must be S, SB or SBM');
end
end
