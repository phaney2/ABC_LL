%PLOT_ABC  ABC trilayer graphene Landau-level density of states.
%
%   Everything you need to change is in the SETTINGS block below.
%
%   datatoplot = 'Edos'  DOS versus energy (Y) and magnetic field (X),
%                        at one interlayer potential. The Landau fan.
%                'ndos'  DOS at the Fermi level versus gate filling (X)
%                        and magnetic field (Y).
%
%   model      = '6band'          full model, gamma0..gamma4 and delta
%                '6band_minimal'  gamma0 and gamma1 only
%                '2band'          effective two-band (A1, B3) model
%
%   Data comes from  python3 scripts/export_for_matlab.py  (set MODEL).
%
%   For 'Edos' the DOS is built here from raw Landau levels, so the
%   broadening below is live. For 'ndos' it was applied on the Python side
%   (finding E_F needs a linewidth), so GAMMA0 is ignored and the value
%   actually used is printed and shown in the title.

clear; close all;

% ======================================================================
%                              SETTINGS
% ======================================================================

datatoplot = 'ndos';        % 'Edos' | 'ndos'
model      = '6band';       % '6band' | '6band_minimal' | '2band'

GAMMA0     = 1.0e-3;        % broadening at 1 T, in eV. 'Edos' ONLY -- that
                            % path builds the DOS here from raw levels, so
                            % this is live.
MAP_GAMMA0 = 2.0e-4;        % 'ndos' ONLY: picks which pre-broadened map to
                            % load. ndos data is broadened in Python when
                            % E_F is solved, so GAMMA0 above cannot sharpen
                            % it -- this selects the file instead. Make a
                            % new one with, in cmd.exe:
                            %   set GAMMA0=0.0002 && set SKIP_LEVELS=1 && ^
                            %   python3 scripts/export_for_matlab.py 6band
SCALING    = 'sqrtB';       % 'sqrtB' -> gamma = GAMMA0*sqrt(B) | 'const'
LINESHAPE  = 'gaussian';    % 'gaussian' | 'lorentzian'

DU_PLOT    = 0.0;           % 'Edos': interlayer potential to show, in eV
E_LIM      = [];            % 'Edos': energy axis, meV. [] = from the file
B_LIM      = [];            % both: field range, T. [] = from the file

NORMALISE  = true;          % 'ndos': divide each field by its own median.
                            % The oscillations ride on a large smooth
                            % background, so without this the fan is
                            % invisible.

% 'ndos': which constant-Landau-filling lines to draw.
%
% The counting is specific to ABC trilayer. Its lowest Landau level is
% 3-fold degenerate in orbital index (n = 0, 1, 2), so with valley and spin
% the zero-mode multiplet holds 3*2*2 = 12 states and straddles charge
% neutrality, spanning nu = -6 to +6. The first gap is therefore at nu = +-6,
% and every level above it is the usual 4-fold (valley x spin):
%
%       nu = +-6, +-10, +-14, +-18, ...
%
% Not plain multiples of 4. Note that a displacement field lifts the valley
% degeneracy, so at finite dU extra gaps open at even nu between these.
NU_LINES   = [-(6:4:30), 6:4:30];
LOGSCALE   = true;          % 'Edos': log colour scale
DECADES    = 4;             % 'Edos': log floor, decades below the peak

% ======================================================================

here = fileparts(mfilename('fullpath'));
tags = struct('x6band', 'six_band', 'x6band_minimal', 'six_band_minimal', ...
              'x2band', 'two_band_minimal');   % see plot_abc_slider.m:
                             % the extended two-band model's dU correction
                             % breaks down at finite Landau index
if ~isfield(tags, ['x' model])
    error('abc:model', 'model must be 6band, 6band_minimal or 2band');
end
tag = tags.(['x' model]);

switch lower(datatoplot)
    case 'edos', f = fullfile(here, '..', 'data', ['levels_' tag '.mat']);
    case 'ndos', f = fullfile(here, '..', 'data', ...
                   sprintf('dosmap_%s_g%gmeV.mat', tag, MAP_GAMMA0*1e3));
    otherwise,   error('abc:what', 'datatoplot must be Edos or ndos');
end
if ~isfile(f)
    error('abc:nodata', ['%s not found.\nGenerate it with:\n' ...
        '    python3 scripts/export_for_matlab.py %s'], f, tag);
end
S = load(f);
fprintf('%s   %s   %s\n', datatoplot, model, f);

NPHI_1T = 2.417989242e14;   % e/h, so n_phi = NPHI_1T*B  [m^-2]
G_SPIN  = 2;

switch lower(datatoplot)
% ---------------------------------------------------------------- Edos
case 'edos'
    B = S.B_T(:);
    [~, iu] = min(abs(S.dU_eV - DU_PLOT));
    if isempty(B_LIM), B_LIM = [min(B) max(B)]; end
    keep = find(B >= B_LIM(1) & B <= B_LIM(2));
    B = B(keep);
    if isempty(E_LIM), E_LIM = [-1 1]*S.energy_window_eV*1e3; end

    % The grid must resolve gamma at the LOWEST field, where gamma ~ sqrt(B)
    % is smallest. Too coarse and the levels alias into dashes that read as
    % structure.
    gmin = gamma_of(GAMMA0, SCALING, min(B));
    nE   = ceil(diff(E_LIM)/(gmin*1e3/2)) + 1;
    E    = linspace(E_LIM(1), E_LIM(2), nE).' * 1e-3;
    fprintf('dU = %.1f meV,  B = %.2f-%.2f T,  %d energy points\n', ...
            S.dU_meV(iu), B(1), B(end), nE);
    fprintf('gamma = %.3f meV at %.2f T (CNP at %.3f meV)\n', ...
            gmin*1e3, min(B), S.cnp_meV(iu));

    D = zeros(nE, numel(B));
    for a = 1:numel(B)
        i = keep(a);
        n  = S.n_levels(i, iu);
        lv = reshape(S.energy_eV(i, iu, 1:n), 1, []);
        g  = gamma_of(GAMMA0, SCALING, B(a));
        D(:, a) = G_SPIN * NPHI_1T * B(a) * ...
                  sum(lineshape(E - lv, g, LINESHAPE), 2);
    end
    D = D / 1e4;                                % cm^-2 eV^-1

    figure('Color', 'w', 'Position', [80 80 900 620]);
    imagesc(B, E*1e3, D); set(gca, 'YDir', 'normal');
    if LOGSCALE
        set(gca, 'ColorScale', 'log');
        clim([max(D(:))*10^(-DECADES), max(D(:))]);
    end
    colormap(parula); cb = colorbar;
    cb.Label.String = 'DOS  (cm^{-2} eV^{-1})';
    yline(S.cnp_meV(iu), 'w-', 'Alpha', 0.5);
    xlabel('magnetic field  B  (T)');
    ylabel('energy  E  (meV)');
    title([model ',  \DeltaU = ' num2str(S.dU_meV(iu), '%.0f') ' meV,  ' ...
           '\gamma = ' num2str(GAMMA0*1e3, '%.2f') ' meV' ...
           scaling_tex(SCALING) ', ' LINESHAPE]);
    box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

% ---------------------------------------------------------------- ndos
case 'ndos'
    B  = S.B_T(:);
    fl = S.filling_moire(:);          % states per moire cell, linear in Vg
    if isempty(B_LIM), B_LIM = [min(B) max(B)]; end
    keep = B >= B_LIM(1) & B <= B_LIM(2);
    B = B(keep);
    D = S.dos_cm2_eV(keep, :);
    fprintf('B = %.2f-%.2f T,  filling %.2f to %.2f,  dU %.0f to %.0f meV\n', ...
            B(1), B(end), fl(1), fl(end), S.dU_meV(1), S.dU_meV(end));
    fprintf('broadening fixed in the data: %s, gamma0 = %.3f meV, %s\n', ...
            S.broadening.lineshape, S.broadening.gamma0_eV*1e3, ...
            S.broadening.scaling);

    lab = 'DOS at E_F  (cm^{-2} eV^{-1})';
    if NORMALISE
        D = D ./ median(D, 2);
        lab = 'DOS at E_F / median at that field';
    end

    figure('Color', 'w', 'Position', [80 80 900 620]);
    imagesc(fl, B, D); set(gca, 'YDir', 'normal');
    clim(pct(D, [2 98]));
    colormap(parula); cb = colorbar; cb.Label.String = lab;
    hold on;
    % constant Landau filling: lines through the origin, since a level
    % holds e*B/h states whatever the Hamiltonian does (Streda)
    A = fl(end) / (S.n_m2(end) + eps);
    for v = NU_LINES
        x = v*NPHI_1T*B*A;
        plot(x, B, '-', 'Color', [1 1 1 0.4], 'LineWidth', 0.6);
        k = find(abs(x) <= max(abs(fl)), 1, 'last');
        if ~isempty(k)
            text(x(k), B(k), sprintf('%+d', v), 'Color', 'w', ...
                 'FontSize', 6.5, 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'bottom');
        end
    end
    xline(0, 'w-', 'Alpha', 0.4);
    xlim([min(fl) max(fl)]); ylim([min(B) max(B)]);
    xlabel('gate filling  (states per moire cell)');
    ylabel('magnetic field  B  (T)');
    title([model ':  DOS at E_F.   \gamma = ' ...
           num2str(S.broadening.gamma0_eV*1e3, '%.2f') ' meV\times\surd(B/1T)' ...
           '.   white: constant Landau \nu']);
    box on; set(gca, 'Layer', 'top', 'TickDir', 'out');
end

% ======================================================================
function g = gamma_of(g0, scaling, B)
switch lower(scaling)
    case 'sqrtb', g = g0 .* sqrt(B);
    case 'const', g = g0 .* ones(size(B));
    otherwise,    error('abc:scaling', 'SCALING must be sqrtB or const');
end
end

function y = lineshape(x, g, shape)
switch lower(shape)
    case 'gaussian',   y = exp(-0.5*(x./g).^2) ./ (g*sqrt(2*pi));
    case 'lorentzian', y = g ./ (pi*(x.^2 + g.^2));
    otherwise, error('abc:shape', 'LINESHAPE must be gaussian or lorentzian');
end
end

function s = scaling_tex(scaling)
if strcmpi(scaling, 'sqrtb'), s = '\times\surd(B/1T)'; else, s = ''; end
end

function lim = pct(X, p)
% percentile without the Statistics Toolbox
v = sort(X(isfinite(X)));
if isempty(v), lim = [0 1]; return; end
i = max(1, min(numel(v), round(p/100 * numel(v))));
lim = [v(i(1)), v(i(2))];
if lim(2) <= lim(1), lim = [v(1), v(end)+eps]; end
end
