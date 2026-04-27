function [Kf, Bf, Pf] = build_helmholtz_filter_Q4(nodes, elements, r)
%BUILD_HELMHOLTZ_FILTER_Q4
%   FE-based Helmholtz density filter for a 2D QUAD4 mesh:
%
%       -r^2 * Laplacian(q) + q = rho
%
%   with homogeneous Neumann boundary conditions.
%
%   INPUT
%   -----
%   nodes    : [nNodes x 2] nodal coordinates
%   elements : [nElem  x 4] QUAD4 connectivity
%   r        : PHYSICAL Helmholtz length scale
%
%   OUTPUT
%   ------
%   Kf : [nNodes x nNodes] Helmholtz FE matrix
%   Bf : [nNodes x nElem ] maps element densities -> Helmholtz RHS
%   Pf : [nElem  x nNodes] maps nodal filtered field -> element-centered field
%
%   Notes
%   -----
%   - Bf assembles the element load vector for piecewise constant element
%     densities.
%   - Pf evaluates the nodal Helmholtz solution at the element center.
%   - This implementation is ONLY good for standard structured QUAD4 meshes.

nNodes = size(nodes, 1);
nElem  = size(elements, 1);

I_K = zeros(16 * nElem, 1);
J_K = zeros(16 * nElem, 1);
V_K = zeros(16 * nElem, 1);

I_B = zeros(4 * nElem, 1);
J_B = zeros(4 * nElem, 1);
V_B = zeros(4 * nElem, 1);

I_P = zeros(4 * nElem, 1);
J_P = zeros(4 * nElem, 1);
V_P = zeros(4 * nElem, 1);

% 2x2 Gauss rule
gp = [-1/sqrt(3),  1/sqrt(3)];
gw = [1, 1];

ptrK = 0;
ptrB = 0;
ptrP = 0;

for e = 1:nElem
    enodes = elements(e, :);
    xe = nodes(enodes, 1);
    ye = nodes(enodes, 2);

    Ke = zeros(4, 4);
    Fe = zeros(4, 1);

    for ii = 1:2
        xi = gp(ii);
        wx = gw(ii);

        for jj = 1:2
            eta = gp(jj);
            wy = gw(jj);

            % Shape functions
            N = 0.25 * [ ...
                (1 - xi) * (1 - eta), ...
                (1 + xi) * (1 - eta), ...
                (1 + xi) * (1 + eta), ...
                (1 - xi) * (1 + eta)];

            % dN/dxi and dN/deta
            dN_dxi = 0.25 * [ ...
                -(1 - eta), ...
                 (1 - eta), ...
                 (1 + eta), ...
                -(1 + eta)];

            dN_deta = 0.25 * [ ...
                -(1 - xi), ...
                -(1 + xi), ...
                 (1 + xi), ...
                 (1 - xi)];

            % Jacobian
            J = [dN_dxi; dN_deta] * [xe ye];
            detJ = det(J);
            invJ = inv(J);

            % Gradients wrt x,y
            dN_dxdy = invJ * [dN_dxi; dN_deta];
            Bgrad = dN_dxdy;  % [2 x 4]

            w = wx * wy * detJ;

            % Helmholtz FE matrix:
            % \int (r^2 grad(N)^T grad(N) + N^T N) dOmega
            Ke = Ke + (r^2 * (Bgrad' * Bgrad) + (N' * N)) * w;

            % RHS vector for constant rho_e:
            % \int N^T * rho_e dOmega
            Fe = Fe + N' * w;
        end
    end

    % Assemble Kf
    [iiLoc, jjLoc] = ndgrid(enodes, enodes);
    idxK = ptrK + (1:16);
    I_K(idxK) = iiLoc(:);
    J_K(idxK) = jjLoc(:);
    V_K(idxK) = Ke(:);
    ptrK = ptrK + 16;

    % Assemble Bf
    idxB = ptrB + (1:4);
    I_B(idxB) = enodes(:);
    J_B(idxB) = e * ones(4,1);
    V_B(idxB) = Fe(:);
    ptrB = ptrB + 4;

    % Assemble Pf at element center (xi = 0, eta = 0)
    idxP = ptrP + (1:4);
    I_P(idxP) = e * ones(4,1);
    J_P(idxP) = enodes(:);
    V_P(idxP) = 0.25 * ones(4,1);
    ptrP = ptrP + 4;
end

Kf = sparse(I_K, J_K, V_K, nNodes, nNodes);
Bf = sparse(I_B, J_B, V_B, nNodes, nElem);
Pf = sparse(I_P, J_P, V_P, nElem, nNodes);
end