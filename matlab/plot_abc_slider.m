%PLOT_ABC_SLIDER  Landau fan DOS(E, B) with a slider over displacement field.
%
%   Full six-band ABC trilayer. Energy on Y, magnetic field on X, and a
%   slider that steps the interlayer potential dU through the values in the
%   data file. The title reports the gate voltage and the gate filling that
%   go with the current dU.
%
%   The DOS is rebuilt from raw Landau levels each time the slider moves, so
%   the broadening below is live.
%
%   Data:  python3 scripts/export_for_matlab.py six_band

function plot_abc_slider()
close all;

% ======================================================================
%                              SETTINGS
% ======================================================================

MODELS     = {'6band', '6band_minimal', '2band'};  % toggle cycles these
model      = '6band';       % which one to start on

GAMMA0     = 2.0e-4;        % broadening at 1 T, in eV
SCALING    = 'sqrtB';       % 'sqrtB' -> gamma = GAMMA0*sqrt(B) | 'const'
LINESHAPE  = 'gaussian';    % 'gaussian' | 'lorentzian'

E_LIM      = [-120 120];    % energy axis, meV.  [] = from the file
B_LIM      = [];            % field range, T.    [] = from the file
N_E        = 1200;          % energy grid points. Kept modest so the slider
                            % stays responsive; raise for a final figure.
LOGSCALE   = true;
DECADES    = 4;             % log floor, decades below the peak

% Gate geometry, used ONLY to convert density into a gate voltage for the
% title. Nothing else in the model depends on it -- the Hamiltonian is
% driven by dU, and the density comes from the lever arm stored in the data
% file. Defaults are a 300 nm SiO2 back gate.
GATE_D     = 300e-9;        % dielectric thickness, m
GATE_EPS   = 3.9;           % relative permittivity

% Interlayer screening. The lever arm stored in the data file is the bare
% electrostatic one: n = kappa*dU with 4 states per moire cell at
% dU = 200 meV. That ignores the charge the layers themselves redistribute,
% which opposes the applied drop. Self-consistent Thomas-Fermi with the
% computed zero-field DOS puts the suppression at roughly 3-4, so a given dU
% needs that much more gate. Set to 1 for the unscreened lever arm.
%
% This rescales ONLY the reported Vg, filling and density -- the Hamiltonian
% is driven by dU directly, so no spectrum changes, just its gate label.
SCREENING  = 3.5;

% ======================================================================

here = fileparts(mfilename('fullpath'));
% '2band' maps to the MINIMAL two-band model deliberately. The extended one
% (abc_ll's 'two_band', which adds the dU gap renormalisation) is unusable at
% these Landau indices: that correction is second-order Schrieffer-Wolff, so
% under minimal substitution it scales as omega^2*n/(2*gamma1^2), about 0.054
% per index at 12 T. It cancels the leading term near n = 18 and changes sign
% beyond it, drifting the two diagonals apart by ~4 meV per index and
% thinning the fan by ~40%. Measured in 70-120 meV at 12 T, dU = 150 meV:
% Onsager predicts 7.2 levels, minimal gives 7, extended gives 4.
tags = struct('x6band','six_band','x6band_minimal','six_band_minimal', ...
              'x2band','two_band_minimal');

% Load every model up front, so the toggle is instant and a missing file is
% reported now rather than halfway through a session.
nM = numel(MODELS);
DAT = struct('name', MODELS, 'tag', [], 'S', [], 'Z', []);
for q = 1:nM
    tg = tags.(['x' MODELS{q}]);
    DAT(q).tag = tg;
    fl = fullfile(here, '..', 'data', ['levels_' tg '.mat']);
    fz = fullfile(here, '..', 'data', ['zerofield_' tg '.mat']);
    if ~isfile(fl)
        error('abc:nodata', ['%s not found.\nGenerate it with:\n' ...
            '    python3 scripts/export_for_matlab.py %s'], fl, tg);
    end
    if ~isfile(fz)
        error('abc:nozf', ['%s not found.\nGenerate it with:\n' ...
            '    python3 scripts/export_zerofield.py %s'], fz, tg);
    end
    DAT(q).S = load(fl);
    DAT(q).Z = load(fz);
end

% Every model must share one dU grid, otherwise the slider index means
% something different in each and the toggle would silently compare
% different physical points.
for q = 2:nM
    if numel(DAT(q).S.dU_eV) ~= numel(DAT(1).S.dU_eV) || ...
       max(abs(DAT(q).S.dU_eV(:) - DAT(1).S.dU_eV(:))) > 1e-9
        error('abc:dugrid', ['%s and %s have different dU grids (%d vs ' ...
            '%d values). Regenerate both with the same NDU.'], ...
            DAT(q).name, DAT(1).name, numel(DAT(q).S.dU_eV), ...
            numel(DAT(1).S.dU_eV));
    end
end

im0 = find(strcmp(MODELS, model), 1);
if isempty(im0), im0 = 1; end
iM  = im0;                       % currently displayed model
S   = DAT(iM).S;
tag = DAT(iM).tag;

EPS0 = 8.8541878128e-12;  QE = 1.602176634e-19;
NPHI_1T = 2.417989242e14; G_SPIN = 2;

B = S.B_T(:);
if isempty(B_LIM), B_LIM = [min(B) max(B)]; end
kb = find(B >= B_LIM(1) & B <= B_LIM(2));
B  = B(kb);
if isempty(E_LIM), E_LIM = [-1 1]*S.energy_window_eV*1e3; end
E  = linspace(E_LIM(1), E_LIM(2), N_E).' * 1e-3;

du   = S.dU_eV(:);
nDU  = numel(du);
[~, i0] = min(abs(du));              % start at dU = 0

% density -> gate voltage. C = eps0*eps_r/d, and V = e*n/C.
if isfield(S, 'n_m2')
    n_m2 = SCREENING * S.n_m2(:);
else
    warning('abc:nogate', ['this file predates the gate export, so Vg and ' ...
        'filling are unavailable; regenerate it to get them']);
    n_m2 = nan(nDU, 1);
end
Cgeo = EPS0*GATE_EPS/GATE_D;
Vg   = QE * n_m2 / Cgeo;
if isfield(S, 'filling_moire'), fill = SCREENING * S.filling_moire(:);
else,                           fill = nan(nDU, 1); end

fprintf('%s,  %d dU values from %.0f to %.0f meV\n', ...
        S.model, nDU, du(1)*1e3, du(end)*1e3);
fprintf('gate: %.0f nm, eps_r = %.1f  ->  %.1f V at dU = %.0f meV\n', ...
        GATE_D*1e9, GATE_EPS, max(Vg), max(du)*1e3);
fprintf('broadening: %s, gamma0 = %.3f meV, %s\n', LINESHAPE, GAMMA0*1e3, SCALING);

Z = DAT(iM).Z;
if numel(Z.dU_eV) ~= nDU || max(abs(Z.dU_eV(:) - du)) > 1e-9
    error('abc:dumismatch', ['the zero-field file has a different dU grid ' ...
        '(%d values) from the levels file (%d). Regenerate both with the ' ...
        'same NDU.'], numel(Z.dU_eV), nDU);
end

% ---------------------------------------------------------------- figure
fig = figure('Color', 'w', 'Position', [50 40 1280 900]);
tl  = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
tl.OuterPosition = [0 0.10 1 0.90];     % leave room for the slider

% Row 1: zero-field band structure and zero-field DOS.
% Row 2: the Landau-level DOS, total and projected on the top layer.
% All four share the energy axis, so they line up vertically.
axB = nexttile(tl);                      % (1,1) bands
% Allocate for the model with the most bands: two_band has 2 where the
% six-band models have 6, and the toggle must not outrun the line handles.
maxNB = max(arrayfun(@(d) size(d.Z.bands_eV, 3), DAT));
hB  = plot(axB, Z.hbarv_kx_eV, nan(numel(Z.hbarv_kx_eV), maxNB));
set(hB, 'LineWidth', 1.1);
xlabel(axB, 'hv k_x  (eV)   [cut through K at k_y = 0]');
ylabel(axB, 'energy  E  (meV)');
title(axB, 'zero-field bands');
box(axB,'on'); set(axB,'TickDir','out'); grid(axB,'on'); hold(axB,'on');
cnpB = yline(axB, 0, 'k--', 'Alpha', 0.5);

axZ = nexttile(tl);                      % (1,2) zero-field DOS
hZ  = plot(axZ, nan(numel(Z.E_meV),1), Z.E_meV, 'LineWidth', 1.1);
xlabel(axZ, 'zero-field DOS  (cm^{-2} eV^{-1})');
title(axZ, 'zero-field DOS');
box(axZ,'on'); set(axZ,'TickDir','out'); grid(axZ,'on'); hold(axZ,'on');
cnpZ = yline(axZ, 0, 'k--', 'Alpha', 0.5);

names = {'Landau DOS: total', 'Landau DOS: top layer'};
ax = gobjects(1, 2); im = gobjects(1, 2); cnpline = gobjects(1, 2);
for q = 1:2
    ax(q) = nexttile(tl);
    im(q) = imagesc(ax(q), B, E*1e3, zeros(N_E, numel(B)));
    set(ax(q), 'YDir', 'normal'); colormap(ax(q), parula);
    xlabel(ax(q), 'magnetic field  B  (T)');
    if q == 1, ylabel(ax(q), 'energy  E  (meV)'); end
    title(ax(q), names{q});
    box(ax(q), 'on'); set(ax(q), 'Layer', 'top', 'TickDir', 'out');
    hold(ax(q), 'on');
    cnpline(q) = yline(ax(q), 0, 'w-', 'Alpha', 0.5);
end
% One shared colourbar and one shared scale, so the two maps are directly
% comparable: the top-layer DOS is a component of the total, never larger,
% and reading that off by eye only works on a common scale.
cb = colorbar(ax(2)); cb.Label.String = 'DOS  (cm^{-2} eV^{-1})';
set([axB axZ ax], 'YLim', E_LIM);

sld = uicontrol(fig, 'Style', 'slider', 'Units', 'normalized', ...
    'Position', [0.10 0.055 0.78 0.04], ...
    'Min', 1, 'Max', nDU, 'Value', i0, ...
    'SliderStep', [1/(nDU-1), max(2/(nDU-1), 0.1)]);
uicontrol(fig, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.10 0.012 0.52 0.03], 'BackgroundColor', 'w', ...
    ... % a uicontrol String is literal -- no TeX, so spell the symbol out
    'String', 'interlayer potential  dU   (drag, or click the trough)');

% Model toggle. Both datasets are already in memory, so switching is just a
% redraw -- the slider position, and so the physical dU, is preserved.
uicontrol(fig, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.655 0.010 0.07 0.028], 'BackgroundColor', 'w', ...
    'HorizontalAlignment', 'right', 'String', 'model:');
pop = uicontrol(fig, 'Style', 'popupmenu', 'Units', 'normalized', ...
    'Position', [0.73 0.012 0.16 0.03], ...
    'String', MODELS, 'Value', iM, ...
    'Callback', @(src,~) switchmodel(src.Value));

% redraw on release and during drag
sld.Callback = @(src,~) redraw(round(src.Value));
addlistener(sld, 'ContinuousValueChange', @(src,~) redraw(round(src.Value)));

redraw(i0);

    function switchmodel(q)
        iM = max(1, min(nM, q));
        S  = DAT(iM).S;
        Z  = DAT(iM).Z;
        % the B grid can differ between files, so re-derive the selection
        Ball = S.B_T(:);
        kb   = find(Ball >= B_LIM(1) & Ball <= B_LIM(2));
        B    = Ball(kb);
        set(im, 'XData', B);
        set([ax(1) ax(2)], 'XLim', [min(B) max(B)]);
        if isfield(S, 'n_m2'), n_m2 = SCREENING * S.n_m2(:);
        else,                  n_m2 = nan(nDU,1); end
        Vg = QE * n_m2 / Cgeo;
        if isfield(S, 'filling_moire'), fill = SCREENING * S.filling_moire(:);
        else,                           fill = nan(nDU,1); end
        redraw(round(sld.Value));
    end

    function redraw(k)
        k = max(1, min(nDU, k));
        D    = zeros(N_E, numel(B));    % total
        Dtop = zeros(N_E, numel(B));    % projected onto layer 3 (top)
        for a = 1:numel(B)
            i  = kb(a);
            n  = S.n_levels(i, k);
            lv = reshape(S.energy_eV(i, k, 1:n), 1, []);
            w3 = reshape(S.layer_weight(i, k, 1:n, 3), [], 1);
            switch lower(SCALING)
                case 'sqrtb', g = GAMMA0*sqrt(B(a));
                otherwise,    g = GAMMA0;
            end
            x = E - lv;
            switch lower(LINESHAPE)
                case 'gaussian',   sh = exp(-0.5*(x./g).^2)./(g*sqrt(2*pi));
                case 'lorentzian', sh = g./(pi*(x.^2 + g.^2));
            end
            deg = G_SPIN * NPHI_1T * B(a);
            D(:, a)    = deg * sum(sh, 2);
            Dtop(:, a) = deg * (sh * w3);
        end
        D = D / 1e4;  Dtop = Dtop / 1e4;

        set(im(1), 'CData', D);
        set(im(2), 'CData', Dtop);
        dmax = max(D(:));                       % common scale, from the total
        for q = 1:2
            if LOGSCALE
                set(ax(q), 'ColorScale', 'log');
                clim(ax(q), [dmax*10^(-DECADES), max(dmax, eps)]);
            else
                clim(ax(q), [0 dmax]);
            end
            set(cnpline(q), 'Value', S.cnp_meV(k));
        end

        % row 1: zero-field bands and DOS at the same dU
        bk = squeeze(Z.bands_meV(k, :, :));
        for q = 1:numel(hB)
            if q <= size(bk, 2)
                set(hB(q), 'XData', Z.hbarv_kx_eV, 'YData', bk(:, q));
            else
                set(hB(q), 'XData', nan, 'YData', nan);   % model has fewer
            end
        end
        set(hZ, 'XData', Z.dos_cm2_eV(k, :));
        set(cnpB, 'Value', Z.cnp_meV(k));
        set(cnpZ, 'Value', Z.cnp_meV(k));
        zmax = max(Z.dos_cm2_eV(k, :));
        if zmax > 0, xlim(axZ, [0 1.05*zmax]); end
        title(tl, sprintf(['%s:  \\DeltaU = %.1f meV,   V_g = %.1f V,   ' ...
            'filling = %.2f / cell   (n = %.3g cm^{-2})\n' ...
            '\\gamma = %.2f meV%s, %s.   lever arm \\times%.2g for screening'], ...
            strrep(MODELS{iM},'_','\_'), du(k)*1e3, Vg(k), fill(k), n_m2(k)/1e4, ...
            GAMMA0*1e3, tex_scaling(SCALING), LINESHAPE, SCREENING));
        drawnow limitrate;
    end
end

function s = tex_scaling(scaling)
if strcmpi(scaling, 'sqrtb'), s = '\times\surd(B/1T)'; else, s = ''; end
end
