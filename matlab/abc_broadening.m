function br = abc_broadening()
%ABC_BROADENING  Landau-level broadening model -- EDIT THIS FILE.
%
%   This is the single place the linewidth is defined.  Both plotting
%   scripts call it, so changing a value here changes every figure.
%
%   The exported .mat files contain *raw Landau levels*, not a pre-broadened
%   density of states, precisely so that this stays a free parameter.
%
%   Fields:
%     gamma0     linewidth at B = 1 T, in eV
%     scaling    'sqrtB'  -> gamma(B) = gamma0 * sqrt(B)   (disorder standard
%                            for graphene: short-range scatterers)
%                'const'  -> gamma(B) = gamma0
%     lineshape  'gaussian' or 'lorentzian'
%
%   On the lineshape.  gamma0 means the standard deviation for a Gaussian and
%   the half-width at half maximum for a Lorentzian, so a Lorentzian of the
%   same gamma0 looks broader.  Prefer the Gaussian: a Lorentzian's 1/E^2
%   tails mean the DOS *inside a gap* -- which is exactly what the minima of
%   a Landau fan are -- picks up weight from levels arbitrarily far away, and
%   does not converge as you widen the energy range you keep.  The exported
%   window is finite (see energy_window_eV), so Lorentzian gap depths here
%   are only meaningful to a few percent.

% ======================================================================
%                        BROADENING PARAMETERS
% ======================================================================

br.gamma0    = 1.0e-6;        % eV at 1 T   <-- main knob
br.scaling   = 'sqrtB';       % 'sqrtB' | 'const'
br.lineshape = 'gaussian';    % 'gaussian' | 'lorentzian'

% ======================================================================

end
