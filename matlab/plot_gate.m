%PLOT_GATE  Gate calibration: density and interlayer potential vs gate voltage.
%
%   Single back gate, so everything is locked together and linear:
%       n  = Cgeo*Vg/e        from the capacitor
%       dU = n/kappa          from the calibration (4 states per moire cell
%                             at the maximum dU)
%   Left axis is the density, right axis the interlayer potential dU.
%   The two curves lie on top of each other by construction -- the plot is a
%   ruler for converting between the three, not a measurement.

clear; close all;

% ======================================================================
%                              SETTINGS
% ======================================================================

GATE_D   = 300e-9;      % dielectric thickness, m
GATE_EPS = 3.9;         % relative permittivity (SiO2)
DU_MAX   = 0.20;        % eV, interlayer potential at full scale
NPTS     = 201;

% ======================================================================

EPS0 = 8.8541878128e-12;  QE = 1.602176634e-19;

% kappa from the exported gate struct if a data file is around, else the
% moire calibration directly: 4 states per cell of side 9.45 nm at DU_MAX
here = fileparts(mfilename('fullpath'));
f = fullfile(here, '..', 'data', 'levels_six_band.mat');
kappa = [];
if isfile(f)
    S = load(f, 'gate');
    if isfield(S, 'gate') && isfield(S.gate, 'kappa_m2_per_eV')
        kappa  = S.gate.kappa_m2_per_eV;
        DU_MAX = S.gate.du_max_eV;
    end
end
if isempty(kappa)
    lam  = 9.45e-9;                         % moire period with hBN
    area = sqrt(3)/2 * lam^2;               % triangular cell
    kappa = 4 / (area * DU_MAX);
end

du   = linspace(-DU_MAX, DU_MAX, NPTS);     % eV
n_m2 = kappa * du;                          % m^-2
Cgeo = EPS0*GATE_EPS/GATE_D;                % F m^-2
Vg   = QE * n_m2 / Cgeo;                    % V
fill = 4 * du / DU_MAX;                     % states per moire cell

fprintf('kappa = %.4g m^-2 eV^-1,  dU_max = %.0f meV\n', kappa, DU_MAX*1e3);
fprintf('Cgeo  = %.4g F m^-2  (%.0f nm, eps_r = %.1f)\n', Cgeo, GATE_D*1e9, GATE_EPS);
fprintf('full scale: Vg = %+.2f V,  n = %+.3e cm^-2,  filling = %+.1f\n', ...
        Vg(end), n_m2(end)/1e4, fill(end));
fprintf('slope: %.4g cm^-2 per V,  %.3f meV per V\n', ...
        (n_m2(end)/1e4)/Vg(end), du(end)*1e3/Vg(end));

figure('Color', 'w', 'Position', [100 100 760 520]);

yyaxis left
plot(Vg, n_m2/1e4, '-', 'LineWidth', 1.8);
ylabel('carrier density  n  (cm^{-2})');
ylim([min(n_m2) max(n_m2)]/1e4);

yyaxis right
plot(Vg, du*1e3, '--', 'LineWidth', 1.4);
ylabel('interlayer potential  \DeltaU  (meV)');
ylim([min(du) max(du)]*1e3);

xlabel('gate voltage  V_g  (V)');
xlim([min(Vg) max(Vg)]);
grid on; box on; set(gca, 'Layer', 'top', 'TickDir', 'out');
title(sprintf(['single back gate: %.0f nm, \\epsilon_r = %.1f.   ' ...
               '\\kappa = %.3g m^{-2}eV^{-1}.   ' ...
               'full scale = 4 states per moire cell'], ...
              GATE_D*1e9, GATE_EPS, kappa));
legend({'density', '\DeltaU'}, 'Location', 'northwest', 'Box', 'off');
