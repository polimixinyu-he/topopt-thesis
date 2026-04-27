clear; clc; close all;

%% USER SWITCHES
opt.filterType = 'helmholtz';   % 'none' | 'conv' | 'helmholtz'
opt.useFilter = true;
opt.useProjection = true;
opt.useBetaContinuation = true;

opt.beta0 = 10.0; % Important! Previous fails in from this value, cannot be too small!
opt.betaMax = 10240.0;
opt.betaUpdateInterval = 40;

opt.eta = 0.5;                     % projection threshold
opt.volumeFractionTarget = 0.5;    % target for the single projected design
opt.maxIter = 200;
opt.move = 0.2;

if ~opt.useFilter
    opt.filterType = 'none';
end

%% PROBLEM SETTINGS

% Material
E       = 70e9;
rho     = 2700;
nu      = 0.33;
thickness = 0.1;

myMaterial = KirchoffMaterial();
set(myMaterial, 'YOUNGS_MODULUS', E, 'DENSITY', rho, 'POISSONS_RATIO', nu);
myMaterial.PLANE_STRESS = true;
myElementConstructor = @() Quad4Element(thickness, myMaterial);

% Mesh
lx = 1; ly = 0.5;
nelx = 100; nely = 50;
hx = lx / nelx;
hy = ly / nely;
% In papers, assign the length unit. E.g. 1 length unit = 1 mm, etc.

[nodes, elements, nset] = mesh_2Drectangle(lx, ly, nelx, nely, 'QUAD4');
myMesh = Mesh(nodes);
myMesh.create_elements_table(elements, myElementConstructor);

% Boundary conditions
myMesh.set_essential_boundary_condition(nset{1}, 1:2, 0);
myMesh.set_essential_boundary_condition(nset{3}, 1, 0);

% Assembly
myAssembly = Assembly(myMesh);

% Nodal force
F = zeros(myMesh.nDOFs, 1);
nf = find_node(lx, ly, [], nodes);
node_force_dofs = get_index(nf, myMesh.nDOFPerNode);
F(node_force_dofs(2)) = -1e3;
Fc = myAssembly.constrain_vector(F);

% Element centroids
coord = zeros(myMesh.nElements, 2);
for ii = 1:myMesh.nElements
    coord(ii, :) = mean(myMesh.Elements(ii).Object.nodes);
end

% Element stiffness matrix for unit density
Ke = myMesh.Elements(1).Object.tangent_stiffness_and_force(zeros(8,1));

% Area
Ae = myMesh.Elements(1).Object.area;
Atot = Ae * myMesh.nElements;

% Null displacement vector
u0 = zeros(myMesh.nDOFs, 1);

%%  TO INITIALIZATION

dMinSimp = 1e-7;
p = 3;

% -------------------------------------------------------------
% Filter radii:
%   RclassicPhys : classical convolution filter radius [physical length]
%   RclassicElem : classical convolution filter radius [number of elements]
%   rHelm        : Helmholtz PDE filter length [physical length]
%
% Matching used here:
%   rHelm = RclassicPhys / (2*sqrt(3))
% -------------------------------------------------------------

RclassicPhys = 2.0 * max(hx, hy);        % Can be modified
RclassicElem = RclassicPhys / max(hx, hy);

rHelm        = RclassicPhys / (2 * sqrt(3));

switch lower(opt.filterType)
    case 'none'
        % explicit H will not be used, but constructor still builds it
        filterRadius = 1.0;

    case 'conv'
        % explicit convolution filter uses radius in element-index space
        filterRadius = RclassicElem;

    case 'helmholtz'
        % explicit H is not used in Helmholtz mode; pass a harmless placeholder
        filterRadius = 1.0;

    otherwise
        error('Unknown filterType.');
end

to = TopologyOptimization_t([nelx, nely], coord, filterRadius, opt.beta0, opt.eta, dMinSimp, p);

% Built-in switches
to.useFilter = opt.useFilter;
to.useProjection = opt.useProjection;
to.filterType = opt.filterType;

% Legacy single-threshold compatibility
to.eta = opt.eta;

% Helmholtz FE filter
if strcmpi(to.filterType, 'helmholtz')
    to.initialize_helmholtz_from_mesh(nodes, elements, rHelm);
end

% Initial design
to.initialize_density(opt.volumeFractionTarget);

figure();
plot_layout(to.nel, to.d, to.mapFea2To);
title('Initial Layout', 'Interpreter', 'latex');
drawnow;

%% MMA INITIALIZATION
m = 1;
mma = MMA(m, opt.move, to);

% history rows:
%   1: objective/compliance
%   2: volume constraint
%   3: Grey Indicator Mnd [%]
history = NaN(m + 2, opt.maxIter);
densHistory = NaN(to.nElements, opt.maxIter);

figure(); drawnow;

%%  MAIN LOOP
fprintf('\nIteration - Obj - Vol - Grey Indicator [%%] - beta\n');

tStart = tic;
iter = 1;

while iter <= opt.maxIter

    % --------------------------------------------------------
    % Beta continuation
    % --------------------------------------------------------
    if to.useProjection && opt.useBetaContinuation
        if iter > 1 && mod(iter - 1, opt.betaUpdateInterval) == 0
            to.beta = min(2 * to.beta, opt.betaMax);
        end
    end

    % --------------------------------------------------------
    % 1) Filtering
    % --------------------------------------------------------
    rhoF = to.apply_filter(to.d);

    % --------------------------------------------------------
    % 2) Projection: single projected design only
    % --------------------------------------------------------
    rhoP = to.project_field(rhoF, to.eta);

    % --------------------------------------------------------
    % 3) SIMP
    % --------------------------------------------------------
    rhoAs = to.simp_field(rhoP);

    % Store state for compatibility / visualization
    to.d_filt = rhoF;
    to.d_proj = rhoP;
    to.d_simp = rhoAs;

    % Volume measure
    V = Ae * sum(rhoP);

    % Grey Indicator:
    % Mnd = mean(4*rhoBar*(1-rhoBar))*100 [%]
    Mnd = to.grey_indicator(rhoP);

    % --------------------------------------------------------
    % 4) FE analysis on the single projected design
    % --------------------------------------------------------
    [K, ~] = myAssembly.tangent_stiffness_and_force_uniform(u0, 'weights', rhoAs);
    Kc = myAssembly.constrain_matrix(K);

    uc = Kc \ Fc;
    u = myAssembly.unconstrain_vector(uc);

    C = dot(Fc, uc);

    % --------------------------------------------------------
    % 5) Sensitivities
    % --------------------------------------------------------
    dCdrhoP = SensitivityLibrary.compliance(myMesh, u, Ke, to.simp_derivative(rhoP));
    dVdrhoP = Ae * ones(myMesh.nElements, 1);

    dProj = to.project_derivative(rhoF, to.eta);

    dC = to.apply_filter_transpose(dCdrhoP .* dProj);
    dV = to.apply_filter_transpose(dVdrhoP .* dProj);

    % --------------------------------------------------------
    % 6) Print + plot
    % --------------------------------------------------------
    fprintf('\n%4d %16.4e %16.4e %18.4f %10.3f', iter, C, V / Atot, Mnd, to.beta);
    plot_layout(to.nel, rhoP, to.mapFea2To);
    drawnow;

    % --------------------------------------------------------
    % 7) MMA data
    % --------------------------------------------------------
    xval  = to.d;
    f0val = C;
    df0dx = dC(:);

    fval  = V - opt.volumeFractionTarget * Atot;
    dfdx  = dV(:).';

    history(:, iter) = [f0val; fval; Mnd];
    densHistory(:, iter) = rhoP;

    % --------------------------------------------------------
    % 8) Convergence check
    % --------------------------------------------------------
    if iter > 5
        fval_tol = 1e-3;
        if all(fval < fval_tol)
            err = abs(1 - history(1, iter-3:iter-1) / history(1, iter));
            err_tol = 1e-3;
            if all(err < err_tol)
                break
            end
        end
    end

    % --------------------------------------------------------
    % 9) MMA update
    % --------------------------------------------------------
    to.d = mma.optimize(iter, xval, f0val, df0dx, fval, dfdx);

    iter = iter + 1;
end

tElapsed = toc(tStart);
fprintf('\n\nEnd of the optimization.\n');
fprintf('Elapsed time is %f seconds.\n', tElapsed);

% Trim histories
history = history(:, 1:iter-1);
densHistory = densHistory(:, 1:iter-1);

%% FINAL RESULTS

rhoF = to.apply_filter(to.d);
rhoP = to.project_field(rhoF, to.eta);
MndFinal = to.grey_indicator(rhoP);

fprintf('Final Grey Indicator Mnd = %.4f %%.\n', MndFinal);

figure();
plot_history(history, {'Compliance', 'Volume constraint', 'Grey Indicator $M_{nd}$ [\%]'});

figure();
plot_layout(to.nel, rhoP, to.mapFea2To);
title('Optimal Projected Layout', 'Interpreter', 'latex');

create_gif(to.nel, densHistory, 'mapFea2To', to.mapFea2To, ...
    'fileName', 'ComplianceMinimization_FeatureControl');
