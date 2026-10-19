%PLOT_SEMICLASSICAL  Semiclassical Onsager fan over the exact Landau levels.
%
%   Grey lines are the exact spectrum from levels_<model>.mat; coloured
%   markers are the semiclassical fan from semiclassical_<model>.mat, built by
%   feeding an ABC band structure to the orbit-area engine in
%   hofstadter/source_code/semiclassical.
%
%   TERMS selects how much of the semiclassical correction to show:
%     'S'    orbit area only -- bare Onsager
%     'SB'   + Berry phase
%     'SBM'  + orbital magnetic moment   <- use this one
%
%   SBM is the accurate one.  Measured against the exact levels it sits at
%   about 0.1 of a level spacing, against 0.2 for SB and 0.3-0.5 for S: the
%   Berry phase and the orbital moment are both first order in B and both are
%   needed.  S and SB are kept so the size of each correction is visible.
%
%   Generate the data with:
%       python3 scripts/export_semiclassical.py six_band

clear; close all;

% ======================================================================
%                              SETTINGS
% ======================================================================

model   = 'six_band';   % 'six_band' | 'six_band_minimal' | 'two_band_minimal'
TERMS   = 'SBM';        % 'S' | 'SB' | 'SBM'  -- see above
DU_PLOT = 0.10;         % interlayer potential to show, in eV
VALLEY  = 'K';          % 'K' | 'Kp' | 'both'
POCKETS = [0 1];        % which orbits to draw; 0 is the largest

B_LIM   = [];           % field range, T.  [] = whole file
E_LIM   = [-160 160];   % energy range, meV
SHOW_EXACT = true;      % grey reference spectrum

% ======================================================================

here = fileparts(mfilename('fullpath'));
f = fullfile(here, '..', 'data', ['semiclassical_' model '.mat']);
if ~isfile(f)
    error('abc:nodata', ['%s not found.\nGenerate it with:\n' ...
        '    python3 scripts/export_semiclassical.py %s'], f, model);
end
S = load(f);

du = S.dU_eV(:);
[~, iu] = min(abs(du - DU_PLOT));
B = S.Blist(:);
if isempty(B_LIM), B_LIM = [min(B) max(B)]; end

gapless = ~all(S.flux_converged(iu, :));
fprintf('%s   dU = %.0f meV   terms = %s\n', model, du(iu)*1e3, TERMS);
fprintf('Bmultiplier = %g (textbook; the engine''s default of 4 is a\n', ...
        S.onsager_Bmultiplier);
fprintf('   Hofstadter-specific factor that does not apply here)\n');
if gapless
    warning('abc:gapless', ['dU = %.0f meV is gapless: the Berry curvature ' ...
        'is a delta function the\nk-grid cannot see, so SB and SBM carry no ' ...
        'Berry phase and reduce to S.'], du(iu)*1e3);
end

figure('Color', 'w', 'Position', [80 80 940 640]); hold on;

% ------------------------------------------------- exact reference
if SHOW_EXACT
    fe = fullfile(here, '..', 'data', ['levels_' model '.mat']);
    if isfile(fe)
        X = load(fe);
        [~, je] = min(abs(X.dU_eV - du(iu)));
        for a = 1:numel(X.B_T)
            n = X.n_levels(a, je);
            if n == 0, continue; end
            e = reshape(X.energy_eV(a, je, 1:n), 1, []) * 1e3;
            e = e(e >= E_LIM(1) & e <= E_LIM(2));
            plot(X.B_T(a)*ones(size(e)), e, '.', 'Color', [0.72 0.72 0.72], ...
                 'MarkerSize', 4, 'HandleVisibility', 'off');
        end
        hExact = plot(nan, nan, '.', 'Color', [0.72 0.72 0.72], ...
                      'MarkerSize', 12, 'DisplayName', 'exact');
    else
        fprintf('(no %s, skipping the exact reference)\n', fe);
        hExact = [];
    end
else
    hExact = [];
end

% ------------------------------------------------- semiclassical fan
if strcmpi(VALLEY, 'both'), valleys = {'K', 'Kp'}; else, valleys = {VALLEY}; end
cols = [0.00 0.45 0.74; 0.85 0.33 0.10];

names = fieldnames(S);
hSemi = []; labels = {};
for iv = 1:numel(valleys)
    v = valleys{iv};
    for ib = 0:1
        for p = POCKETS
            % matches LL_<v>_b<ib>_p<p>_<TERMS> and its _segN variants
            pat = sprintf('^LL_%s_b%d_p%d_%s(_seg\\d+)?$', v, ib, p, TERMS);
            hit = names(~cellfun(@isempty, regexp(names, pat, 'once')));
            for k = 1:numel(hit)
                A = squeeze(S.(hit{k})(iu, :, :));    % (nB, nmax+1)
                for n = 1:size(A, 2)
                    e = A(:, n);
                    m = isfinite(e) & e >= E_LIM(1) & e <= E_LIM(2) & ...
                        B >= B_LIM(1) & B <= B_LIM(2);
                    if ~any(m), continue; end
                    h = plot(B(m), e(m), '-', 'Color', cols(iv, :), ...
                             'LineWidth', 1.0, 'HandleVisibility', 'off');
                end
            end
            if ~isempty(hit) && ib == 1 && p == POCKETS(1)
                hSemi(end+1) = plot(nan, nan, '-', 'Color', cols(iv, :), ...
                    'LineWidth', 1.5); %#ok<SAGROW>
                labels{end+1} = [v ' semiclassical']; %#ok<SAGROW>
            end
        end
    end
end

xlim(B_LIM); ylim(E_LIM);
xlabel('magnetic field  B  (T)');
ylabel('energy  E  (meV)');
ttl = sprintf('%s:  \\DeltaU = %.0f meV,  %s', strrep(model,'_','\_'), ...
              du(iu)*1e3, terms_label(TERMS));
if gapless, ttl = [ttl '   (gapless: no Berry term)']; end
title(ttl);
if ~isempty([hExact hSemi])
    legend([hExact hSemi], [{'exact'} labels], 'Location', 'northwest', ...
           'Box', 'off');
end
box on; grid on; set(gca, 'Layer', 'top', 'TickDir', 'out');

% ======================================================================
function s = terms_label(t)
switch upper(t)
    case 'S',   s = 'orbit area only';
    case 'SB',  s = 'area + Berry phase';
    case 'SBM', s = 'area + Berry phase + orbital moment';
    otherwise,  error('abc:terms', 'TERMS must be S, SB or SBM');
end
end
