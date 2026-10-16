%PLOT_DOS_FILLING_VS_B  DOS at the Fermi level: filling (X) vs field (Y).
%
%   For each (filling, field) the chemical potential is solved so that the
%   Landau levels hold exactly nu carriers per flux quantum, then the DOS is
%   evaluated there.  Gaps between Landau levels show up as vertical dark
%   stripes at integer nu.
%
%   Energies are absolute and charge neutrality is NOT at E = 0 -- delta and
%   gamma4 put it near +6.3 meV at dU = 0.  The file carries it as cnp_eV and
%   it is used as the reference for the filling.
%
%   Single back gate, so the interlayer potential is not independent:
%   dU = nu*(e*B/h)/kappa.  It is stored as dU_eV (nB x nNu) and is shown in
%   the second figure, because it varies across the plot.
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

datafile = fullfile(here, '..', 'data', 'll_vs_nu.mat');

NU_LIM   = [-14 14];   % filling-factor range to show ([] = whole file)
B_LIM    = [];         % field range in T ([] = whole file)

LOG_SCALE   = true;    % log colour scale
DECADES     = 3;       % log floor, this many decades below the peak. Avoids
                       % stretching the colour map over the near-zero DOS
                       % found deep inside a gap.
SHOW_EXTRAS = true;    % second figure: Fermi level, top-layer fraction, dU
NU_GRID     = -12:4:12;% filling factors to mark with guide lines

br = abc_broadening();          % <-- BROADENING: edit abc_broadening.m
% ======================================================================

t = abc_dos_tools();
S = load(datafile);

nu = S.nu(:);  B = S.B_T(:);
ii = true(size(B));  jj = true(size(nu));
if ~isempty(B_LIM),  ii = B  >= B_LIM(1)  & B  <= B_LIM(2);  end
if ~isempty(NU_LIM), jj = nu >= NU_LIM(1) & nu <= NU_LIM(2); end
iB = find(ii);  iN = find(jj);

fprintf('%s\n', S.model);
fprintf('nu = %.1f to %.1f (%d points),  B = %.2f to %.2f T (%d points)\n', ...
        nu(iN(1)), nu(iN(end)), numel(iN), B(iB(1)), B(iB(end)), numel(iB));
fprintf('broadening: %s, gamma0 = %.3f meV, scaling %s\n', ...
        br.lineshape, br.gamma0*1e3, br.scaling);
fprintf('            gamma = %.2f meV at %.0f T\n', ...
        t.width(br, B(iB(end)))*1e3, B(iB(end)));

nB = numel(iB);  nN = numel(iN);
D   = nan(nB, nN);      % DOS at E_F
MU  = nan(nB, nN);      % Fermi level
TOP = nan(nB, nN);      % fraction of DOS(E_F) on the top layer

for a = 1:nB
    i = iB(a);
    for b = 1:nN
        j = iN(b);
        lv = t.levels(S, i, j);
        mu = t.fermi(br, nu(j), lv, B(i), S.cnp_eV(i, j));
        if isnan(mu), continue; end
        MU(a, b) = mu;
        dl = t.dos(br, mu, lv, B(i), t.weights(S, i, j));   % 1 x 3
        D(a, b)   = sum(dl);
        TOP(a, b) = dl(3) / sum(dl);
    end
end
D = D / 1e4;            % states cm^-2 eV^-1

nFail = sum(isnan(D(:)));
if nFail > 0
    fprintf('%d of %d points had no solvable Fermi level\n', nFail, numel(D));
end

% ---------------------------------------------------------------- plot
figure('Color', 'w', 'Position', [80 80 780 580]);
imagesc(nu(iN), B(iB), D); set(gca, 'YDir', 'normal');
dmax = max(D(:));
if LOG_SCALE
    set(gca, 'ColorScale', 'log');
    clim([dmax*10^(-DECADES), dmax]);
end
colormap(parula);
cb = colorbar; cb.Label.String = 'DOS at E_F  (cm^{-2} eV^{-1})';
hold on;
for v = NU_GRID
    xline(v, 'w-', 'Alpha', 0.35);
end
xlabel('filling factor  \nu');
ylabel('magnetic field  B  (T)');
title(sprintf(['ABC trilayer, DOS at E_F.  Single gate: ' ...
               '\\DeltaU = \\nu(eB/h)/\\kappa.  \\gamma = %.2f meV' ...
               '\\times\\surd(B/1T), %s'], br.gamma0*1e3, br.lineshape));
box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

if SHOW_EXTRAS
    figure('Color', 'w', 'Position', [120 120 1320 420]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    imagesc(nu(iN), B(iB), MU*1e3); set(gca, 'YDir', 'normal');
    colormap(gca, parula); cb = colorbar; cb.Label.String = 'E_F  (meV)';
    xlabel('filling factor  \nu'); ylabel('B  (T)'); title('Fermi level');
    box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

    nexttile;
    imagesc(nu(iN), B(iB), TOP); set(gca, 'YDir', 'normal'); clim([0 1]);
    colormap(gca, parula); cb = colorbar;
    cb.Label.String = 'fraction of DOS(E_F) on top layer';
    xlabel('filling factor  \nu'); title('Top-layer projection');
    box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

    nexttile;
    imagesc(nu(iN), B(iB), S.dU_eV(iB, iN)*1e3); set(gca, 'YDir', 'normal');
    colormap(gca, parula); cb = colorbar; cb.Label.String = '\DeltaU  (meV)';
    xlabel('filling factor  \nu');
    title('Interlayer potential (set by \nu and B)');
    box on; set(gca, 'Layer', 'top', 'TickDir', 'out');

    title(tl, 'Context for the DOS map');
end
