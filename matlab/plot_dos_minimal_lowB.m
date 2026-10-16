%PLOT_DOS_MINIMAL_LOWB  Spectrum DOS(E) vs B, minimal six-band model, dU = 0.
%
%   Low-field Landau fan of ABC trilayer graphene with only gamma0 and
%   gamma1 kept, at zero displacement field.  Energy on Y, field on X.
%
%   Set DU_TAG to pick the dataset: 'dU0' or 'dU200'.
%
%   For this model charge neutrality really is at E = 0, and the spectrum is
%   particle-hole symmetric about it.  Only gamma0 and gamma1 are kept, so
%   the ABC chain stays bipartite, and the layer potential is traceless --
%   chirality combined with layer reversal then maps the spectrum to minus
%   itself (swapping valleys, which are merged here).  That holds at any dU.
%   It is the point of comparison against the full model, where delta and
%   gamma4 break it and put the Dirac points at +6.3 meV instead.
%
%   The broadening lives in abc_broadening.m; edit it there.

clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(here);

% ======================================================================
%                              SETTINGS
% ======================================================================

DU_TAG   = getenv('ABC_DU_TAG');          % override from the shell
if isempty(DU_TAG), DU_TAG = 'dU0'; end   % 'dU0' or 'dU200'
datafile = fullfile(here, '..', 'data', ['ll_minimal_lowB_' DU_TAG '.mat']);

E_LIM   = [];          % meV, energy axis. [] = take it from the file's
                       % energy window, which already differs between the
                       % two cases (the dU = 200 meV run has to reach past
                       % the 163 meV gap).
N_E     = 0;           % energy grid points. 0 = size it automatically to
                       % resolve gamma at the LOWEST field, where
                       % gamma ~ sqrt(B) is smallest. Set a number to force
                       % it; the script warns if that number is too small.
LOG_SCALE = true;
DECADES = 4;           % log floor, this many decades below the peak
B_CUTS  = [0.5 2 5];   % fields for the waterfall panel ([] to skip)
SYM_CHECK = true;      % verify particle-hole symmetry of the spectrum

br = abc_broadening();          % <-- BROADENING: edit abc_broadening.m
% ======================================================================

t = abc_dos_tools();
S = load(datafile);
B = S.B_T(:);
iu = 1;                         % the file holds the single value dU = 0
if isempty(E_LIM)
    E_LIM = [-1 1] * S.energy_window_eV * 1e3;
end
gmin_meV = t.width(br, min(B))*1e3;
N_NEEDED = ceil(diff(E_LIM)/(gmin_meV/2)) + 1;
if N_E <= 0
    N_E = N_NEEDED;
end
E  = linspace(E_LIM(1), E_LIM(2), N_E).' * 1e-3;    % eV
dE = (E(2)-E(1))*1e3;

fprintf('%s,  dU = %.1f meV\n', S.model, S.dU_meV(iu));
fprintf('B = %.2f to %.2f T, %d points\n', B(1), B(end), numel(B));
fprintf('levels in file: %d at %.2f T down to %d at %.1f T\n', ...
        max(S.n_levels(:)), B(1), min(S.n_levels(:)), B(end));
fprintf('broadening: %s, gamma0 = %.3f meV, scaling %s\n', ...
        br.lineshape, br.gamma0*1e3, br.scaling);
gmin = t.width(br, B(1))*1e3;  gmax = t.width(br, B(end))*1e3;
fprintf('            gamma = %.3f meV at %.2f T, %.3f meV at %.1f T\n', ...
        gmin, B(1), gmax, B(end));

% The low-field end is where this plot can lie to you: gamma shrinks as
% sqrt(B) while the levels crowd together, so check the energy grid can
% actually resolve a line before trusting the picture.
fprintf('energy grid: %d points, %.4f meV spacing (need <= %.4f)\n', ...
        N_E, dE, gmin/2);
if dE > gmin/2
    warning('abc:grid', ['energy grid is %.3f meV but gamma is only ' ...
        '%.3f meV at %.2f T: lines are under-sampled and the low-field ' ...
        'end will alias. Raise N_E to >= %d, or raise gamma0.'], ...
        dE, gmin, B(1), N_NEEDED);
end

D = zeros(N_E, numel(B));
for i = 1:numel(B)
    D(:, i) = t.dos(br, E, t.levels(S, i, iu), B(i));
end
D = D / 1e4;        % states cm^-2 eV^-1

% Particle-hole symmetry check.  For the minimal model the only couplings
% are gamma0 and gamma1, and the layer potential is traceless, so chirality
% combined with layer reversal maps the spectrum to minus itself (swapping
% valleys on the way).  Both valleys are merged in the file, so the DOS must
% come out even in E -- at any dU, not just zero.  If this ever fails, a
% term that breaks the bipartite structure has crept in.
if SYM_CHECK
    asym = max(abs(D - flipud(D)), [], 'all') / max(D(:));
    fprintf('particle-hole asymmetry of DOS(E): %.2e (relative)\n', asym);
    if asym > 1e-6
        warning('abc:phsym', ['DOS is not even in E (%.2e). The minimal ' ...
            'model should be particle-hole symmetric about E = 0.'], asym);
    end
end

% ---------------------------------------------------------------- plot
figure('Color', 'w', 'Position', [80 80 900 620]);
imagesc(B, E*1e3, D); set(gca, 'YDir', 'normal');
dmax = max(D(:));
if LOG_SCALE
    set(gca, 'ColorScale', 'log');
    clim([dmax*10^(-DECADES), dmax]);
end
colormap(parula);
cb = colorbar; cb.Label.String = 'DOS  (cm^{-2} eV^{-1})';
xlabel('magnetic field  B  (T)');
ylabel('energy  E  (meV)');
yline(0, 'w-', 'Alpha', 0.5, 'HandleVisibility', 'off');
% built by concatenation, not sprintf: sprintf would try to interpret the
% TeX backslashes (\Delta, \gamma, \times) as C escape sequences
title(['ABC trilayer, ' strrep(S.model, '_', '\_') ...
       ', \DeltaU = ' num2str(S.dU_meV(iu), '%.0f') ' meV.   ' ...
       '\gamma = ' num2str(br.gamma0*1e3, '%.2f') ...
       ' meV\times\surd(B/1T), ' br.lineshape]);
box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

% Waterfall of a few fields: energy on X so the peaks read left to right,
% each trace offset upward so they do not overlap.
if ~isempty(B_CUTS)
    figure('Color', 'w', 'Position', [120 120 980 560]);
    hold on;
    cols = lines(numel(B_CUTS));
    step = 1.25;
    for c = 1:numel(B_CUTS)
        [~, i] = min(abs(B - B_CUTS(c)));
        y = D(:, i) / dmax + (c-1)*step;
        plot(E*1e3, y, 'LineWidth', 0.9, 'Color', cols(c, :));
        text(E_LIM(1) + 0.02*diff(E_LIM), (c-1)*step + 0.95*step, ...
             sprintf('B = %.2f T', B(i)), 'Color', cols(c, :), ...
             'FontWeight', 'bold');
    end
    xline(0, 'k-', 'Alpha', 0.25, 'HandleVisibility', 'off');
    xlim(E_LIM); ylim([-0.1, numel(B_CUTS)*step]);
    set(gca, 'YTick', []);
    xlabel('energy  E  (meV)');
    ylabel('DOS (normalised, offset per trace)');
    title(['DOS(E) at fixed field,  \DeltaU = ' ...
           num2str(S.dU_meV(iu), '%.0f') ' meV']);
    box on; set(gca, 'TickDir', 'out');
end
