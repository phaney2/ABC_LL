%PLOT_DOS_ENERGY_VS_B  Density of states: energy (Y) vs magnetic field (X).
%
%   Builds DOS(E, B) from the raw Landau levels in ll_vs_B.mat, at a chosen
%   interlayer potential.  This is the spectral DOS -- the Landau fan itself,
%   with the levels visible as bright curves -- not the DOS at the Fermi
%   level.  For that, see plot_dos_filling_vs_B.m.
%
%   The broadening lives in abc_broadening.m; edit it there.
%
%   ABC (rhombohedral) trilayer graphene, six-band continuum model,
%   both valleys, spin degeneracy included.

clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(here);

% ======================================================================
%                              SETTINGS
% ======================================================================

datafile = fullfile(here, '..', 'data', 'll_vs_B.mat');

DU_PICK  = 0.00;      % eV, interlayer potential to plot. The file holds
                       % -0.20 -0.10 -0.05 0 0.05 0.10 0.20; the nearest
                       % is used.  dU > 0 is electron doping, dU < 0 holes.

E_LIM    = [-120 120]; % meV, energy axis range
N_E      = 801;        % energy grid points

LOG_SCALE   = true;    % log colour scale (the fan spans decades)
DECADES     = 4;       % log scale floor, this many decades below the peak.
                       % Needed because a Gaussian decays to ~1e-50 between
                       % levels: those values are numerically nonzero but
                       % physically meaningless, and a percentile-based range
                       % would stretch the colour map over ~60 dead decades.
SHOW_LAYERS = true;    % second figure: DOS projected on each layer

br = abc_broadening();          % <-- BROADENING: edit abc_broadening.m
% ======================================================================

t = abc_dos_tools();
S = load(datafile);

[~, iu] = min(abs(S.dU_eV - DU_PICK));
du = S.dU_eV(iu);
B  = S.B_T(:);
E  = linspace(E_LIM(1), E_LIM(2), N_E).' * 1e-3;    % eV

fprintf('%s\n', S.model);
fprintf('dU = %.1f meV   (requested %.1f)\n', du*1e3, DU_PICK*1e3);
fprintf('B  = %.2f to %.2f T, %d points\n', B(1), B(end), numel(B));
fprintf('broadening: %s, gamma0 = %.3f meV, scaling %s\n', ...
        br.lineshape, br.gamma0*1e3, br.scaling);
fprintf('            gamma = %.2f meV at %.0f T\n', ...
        t.width(br, B(end))*1e3, B(end));

D  = zeros(N_E, numel(B));                 % total DOS
DL = zeros(N_E, numel(B), 3);              % per layer
for i = 1:numel(B)
    lv = t.levels(S, i, iu);
    D(:, i) = t.dos(br, E, lv, B(i));
    if SHOW_LAYERS
        DL(:, i, :) = t.dos(br, E, lv, B(i), t.weights(S, i, iu));
    end
end
D  = D  / 1e4;      % states cm^-2 eV^-1
DL = DL / 1e4;

% ---------------------------------------------------------------- plot
figure('Color', 'w', 'Position', [80 80 760 560]);
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
title(sprintf(['ABC trilayer DOS,  \\DeltaU = %.0f meV,  ' ...
               '\\gamma = %.2f meV\\times\\surd(B/1T), %s'], ...
              du*1e3, br.gamma0*1e3, br.lineshape));
yline(0, 'w-', 'Alpha', 0.4);
box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

if SHOW_LAYERS
    names = {'layer 1 (bottom)', 'layer 2 (middle)', 'layer 3 (top)'};
    figure('Color', 'w', 'Position', [120 120 1320 420]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    axL = gobjects(1,3);
    for L = 1:3
        axL(L) = nexttile;
        imagesc(B, E*1e3, DL(:, :, L)); set(gca, 'YDir', 'normal');
        if LOG_SCALE
            set(gca, 'ColorScale', 'log');
            clim([dmax*10^(-DECADES), dmax]);    % common scale across layers
        end
        colormap(parula);
        xlabel('B  (T)');
        if L == 1, ylabel('energy  E  (meV)'); end
        title(names{L});
        yline(0, 'w-', 'Alpha', 0.4);
        box on; set(gca, 'Layer', 'top', 'TickDir', 'out');
    end
    cb = colorbar(axL(3)); cb.Label.String = 'DOS  (cm^{-2} eV^{-1})';
    title(tl, sprintf('Layer-projected DOS,  \\DeltaU = %.0f meV', du*1e3));
end
