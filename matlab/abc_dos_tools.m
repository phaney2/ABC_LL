function t = abc_dos_tools()
%ABC_DOS_TOOLS  Shared routines for building a DOS from raw Landau levels.
%
%   t = abc_dos_tools() returns a struct of function handles:
%
%     t.width(br, B)            linewidth at field B, in eV
%     t.pdf(br, x, g)           normalised lineshape (integrates to 1)
%     t.cdf(br, x, g)           its cumulative integral
%     t.levels(S, i, j)         valid levels at grid point (i,j), NaNs stripped
%     t.weights(S, i, j)        matching layer weights, (n x 3)
%     t.dos(br, E, lv, B, w)    DOS at energies E, optionally layer-projected
%     t.fermi(br, nu, lv, B, e0) chemical potential at filling nu
%
%   Conventions matching the Python side:
%     * energies in eV on an absolute axis.  **Charge neutrality is NOT at
%       E = 0**: delta and gamma4 break the chiral symmetry of the ABC chain
%       and push the low-energy structure to about +6.3 meV at dU = 0.  The
%       exported files carry the neutrality energy as cnp_eV -- always pass
%       it to t.fermi, or every filling will be offset.
%     * each Landau level holds g_spin * e*B/h states per unit area, with
%       g_spin = 2; both valleys are already present in the level list
%     * filling factor nu counts all four spin/valley flavours
%     * DOS comes out in states m^-2 eV^-1

t.NPHI_1T = 2.417989242e14;   % e/h, so n_phi = NPHI_1T * B  [m^-2]
t.G_SPIN  = 2;

t.width   = @width_;
t.pdf     = @pdf_;
t.cdf     = @cdf_;
t.levels  = @levels_;
t.weights = @weights_;
t.dos     = @(br, E, lv, B, varargin) dos_(t, br, E, lv, B, varargin{:});
t.fermi   = @(br, nu, lv, B, e0) fermi_(br, nu, lv, B, e0);
end

% ----------------------------------------------------------------------
function g = width_(br, B)
switch lower(br.scaling)
    case 'sqrtb', g = br.gamma0 .* sqrt(B);
    case 'const', g = br.gamma0 .* ones(size(B));
    otherwise,    error('abc:scaling', 'unknown scaling "%s"', br.scaling);
end
end

function y = pdf_(br, x, g)
switch lower(br.lineshape)
    case 'gaussian',   y = exp(-0.5*(x./g).^2) ./ (g*sqrt(2*pi));
    case 'lorentzian', y = g ./ (pi*(x.^2 + g.^2));
    otherwise, error('abc:lineshape', 'unknown lineshape "%s"', br.lineshape);
end
end

function y = cdf_(br, x, g)
switch lower(br.lineshape)
    case 'gaussian',   y = 0.5*(1 + erf(x./(g*sqrt(2))));
    case 'lorentzian', y = 0.5 + atan(x./g)/pi;
    otherwise, error('abc:lineshape', 'unknown lineshape "%s"', br.lineshape);
end
end

% ----------------------------------------------------------------------
function e = levels_(S, i, j)
%LEVELS_  Valid levels at grid point (i,j).  The arrays are NaN-padded along
%   dimension 3 to the largest level count anywhere on the grid; n_levels
%   records the true count.
n = S.n_levels(i, j);
e = reshape(S.energy_eV(i, j, 1:n), [], 1);
end

function w = weights_(S, i, j)
n = S.n_levels(i, j);
w = reshape(S.layer_weight(i, j, 1:n, :), n, 3);
end

% ----------------------------------------------------------------------
function d = dos_(t, br, E, lv, B, w)
%DOS_  Density of states at energies E (vector), in states m^-2 eV^-1.
%   With a 4th argument w (n x 3 layer weights) the result is n_E x 3,
%   resolved by layer and summing to the total.
g   = width_(br, B);
E   = E(:);
lv  = lv(:);
deg = t.G_SPIN * t.NPHI_1T * B;
shp = pdf_(br, E - lv.', g);            % (nE x nLev)
if nargin < 6 || isempty(w)
    d = deg * sum(shp, 2);
else
    d = deg * (shp * w);                % (nE x 3)
end
end

% ----------------------------------------------------------------------
function mu = fermi_(br, nu, lv, B, e0)
%FERMI_  Chemical potential at filling nu, by bisection.
%
%   n(mu) = g_spin * n_phi * sum_i [ CDF(mu - E_i) - CDF(e0 - E_i) ], and we
%   want n = nu * n_phi, so n_phi drops out entirely:
%
%       sum_i [ CDF(mu - E_i) - CDF(e0 - E_i) ] = nu / g_spin
%
%   e0 is the charge-neutrality energy (cnp_eV in the data file), NOT zero:
%   delta and gamma4 put it near +6.3 meV at dU = 0.  Using zero instead
%   offsets the filling by up to ~1e11 cm^-2.
%
%   n(mu) increases monotonically and vanishes at e0, so sign(mu - e0) =
%   sign(nu) and plain bisection is robust.  Returns NaN if the target
%   filling is outside what the exported energy window can hold.
if nargin < 5 || isempty(e0)
    error('abc:cnp', ['pass the neutrality energy e0 (cnp_eV from the ' ...
                      'data file); it is not zero']);
end
g   = width_(br, B);
lv  = lv(:);
base = cdf_(br, e0 - lv, g);
target = nu / 2;                        % g_spin = 2
f = @(m) sum(cdf_(br, m - lv, g) - base);

lo = min(lv) - 10*g;
hi = max(lv) + 10*g;
if f(lo) > target || f(hi) < target
    mu = NaN;  return
end
for k = 1:200
    mid = 0.5*(lo + hi);
    if f(mid) < target, lo = mid; else, hi = mid; end
    if hi - lo < 1e-10, break; end
end
mu = 0.5*(lo + hi);
end
