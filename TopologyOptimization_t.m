classdef TopologyOptimization_t < handle
    % TopologyOptimization_t: topology optimization helper class.

    properties
        nDim
        nel
        nElements
        coord
        freeElements
        mapFea2To, mapTo2Fea
        d, d_filt, d_proj, d_simp, d_ramp
        radius, H
        beta, eta
        dMinSimp, p
        dMinRamp, q

        % Built-in switches
        useFilter = true
        useProjection = true
        filterType = 'conv'   % 'none' | 'conv' | 'helmholtz'

        % FE-based Helmholtz filter operators
        Kf = []
        Bf = []
        Pf = []
        KfFactor = []   % cached factorization for repeated solves
        rHelm = []      % physical Helmholtz length scale

    end

    methods
        function obj = TopologyOptimization_t(nel, coord, radius, beta, eta, dMinSimp, p, varargin)
            % Constructor.
            %
            % Inputs:
            %   nel      : [1 x nDim] number of elements in each direction
            %   coord    : [nElements x nDim] element centroid coordinates
            %   radius   : explicit convolution filter radius
            %   beta     : projection sharpness parameter
            %   eta      : legacy single-threshold projection parameter
            %   dMinSimp : minimum SIMP density
            %   p        : SIMP penalization
            %
            % Optional:
            %   dMinRamp : minimum RAMP density (default 1e-6)
            %   q        : RAMP parameter (default 4)

            parser = inputParser;
            addOptional(parser, 'dMinRamp', 1e-6);
            addOptional(parser, 'q', 4);
            parse(parser, varargin{:});
            dMinRamp = parser.Results.dMinRamp;
            q = parser.Results.q;

            obj.nDim = length(nel);
            obj.nel = nel(:).';
            obj.nElements = size(coord, 1);
            obj.coord = coord;
            obj.radius = radius;   % explicit convolution filter radius
            obj.beta = beta;
            obj.eta = eta;         % legacy single-threshold projection
            obj.dMinSimp = dMinSimp;
            obj.p = p;
            obj.dMinRamp = dMinRamp;
            obj.q = q;

            % Defaults
            obj.useFilter = true;
            obj.useProjection = true;
            obj.filterType = 'conv';

            % Helmholtz-related data
            obj.Kf = [];
            obj.Bf = [];
            obj.Pf = [];
            obj.KfFactor = [];
            obj.rHelm = [];


            % Free elements indicator
            obj.freeElements = ones(obj.nElements, 1);

            % Density fields
            obj.d      = zeros(obj.nElements, 1);
            obj.d_filt = zeros(obj.nElements, 1);
            obj.d_proj = zeros(obj.nElements, 1);
            obj.d_simp = zeros(obj.nElements, 1);
            obj.d_ramp = zeros(obj.nElements, 1);

            % Map between FEA and TO domains
            [obj.mapFea2To, obj.mapTo2Fea] = map_fea_2_to(obj.coord);

            % Initialize explicit convolution filter H by default
            if obj.nDim == 2
                obj.initialize_filter_2d;
            elseif obj.nDim == 3
                obj.initialize_filter_3d;
            else
                error('The dimension of the domain should be 2 or 3.');
            end
        end

        function initialize_filter_2d(obj)
            % Initialize the explicit convolution filter for 2D domains.
            %
            % Performance notes:
            %   - Neighbor offsets and their unnormalized weights are
            %     precomputed once instead of rebuilding meshgrid arrays for
            %     every element.
            %   - The sparse triplet storage is allocated with a safe upper
            %     bound and trimmed before sparse assembly.
            %   - Non-integer radii are handled by integer offset windows.

            mapElements = reshape(obj.mapFea2To, obj.nel);

            rInd = max(0, ceil(obj.radius));
            [dxGrid, dyGrid] = ndgrid(-rInd:rInd, -rInd:rInd);
            dist = sqrt(dxGrid.^2 + dyGrid.^2);
            inside = dist < obj.radius;

            dxOff = dxGrid(inside);
            dyOff = dyGrid(inside);
            wOff  = obj.radius - dist(inside);

            % Fallback for radius <= 0 or extremely small radii.
            if isempty(wOff)
                obj.H = speye(obj.nElements);
                return;
            end

            nOffsets = numel(wOff);
            nTriplets = obj.nElements * nOffsets;
            iRow = zeros(nTriplets, 1);
            iCol = zeros(nTriplets, 1);
            vals = zeros(nTriplets, 1);

            k = 1;
            for y = 1:obj.nel(2)
                validY = (y + dyOff >= 1) & (y + dyOff <= obj.nel(2));
                for x = 1:obj.nel(1)
                    valid = validY & (x + dxOff >= 1) & (x + dxOff <= obj.nel(1));

                    xN = x + dxOff(valid);
                    yN = y + dyOff(valid);
                    w  = wOff(valid);
                    w  = w ./ sum(w);

                    idx = mapElements(x, y);
                    cols = mapElements(sub2ind(obj.nel, xN, yN));
                    nPositive = numel(cols);

                    range = k:(k + nPositive - 1);
                    iRow(range) = idx;
                    iCol(range) = cols;
                    vals(range) = w;
                    k = k + nPositive;
                end
            end

            iRow = iRow(1:k-1);
            iCol = iCol(1:k-1);
            vals = vals(1:k-1);

            obj.H = sparse(iRow, iCol, vals, obj.nElements, obj.nElements);
        end

        function initialize_filter_3d(obj)
            % Initialize the explicit convolution filter for 3D domains.
            %
            % This optimized version mirrors initialize_filter_2d: the
            % candidate offset stencil is built once, then clipped near
            % boundaries for each element.  ndgrid is used so offset ordering
            % matches mapElements(x,y,z).

            mapElements = reshape(obj.mapFea2To, obj.nel);

            rInd = max(0, ceil(obj.radius));
            [dxGrid, dyGrid, dzGrid] = ndgrid(-rInd:rInd, -rInd:rInd, -rInd:rInd);
            dist = sqrt(dxGrid.^2 + dyGrid.^2 + dzGrid.^2);
            inside = dist < obj.radius;

            dxOff = dxGrid(inside);
            dyOff = dyGrid(inside);
            dzOff = dzGrid(inside);
            wOff  = obj.radius - dist(inside);

            if isempty(wOff)
                obj.H = speye(obj.nElements);
                return;
            end

            nOffsets = numel(wOff);
            nTriplets = obj.nElements * nOffsets;
            iRow = zeros(nTriplets, 1);
            iCol = zeros(nTriplets, 1);
            vals = zeros(nTriplets, 1);

            k = 1;
            for z = 1:obj.nel(3)
                validZ = (z + dzOff >= 1) & (z + dzOff <= obj.nel(3));
                for y = 1:obj.nel(2)
                    validYZ = validZ & (y + dyOff >= 1) & (y + dyOff <= obj.nel(2));
                    for x = 1:obj.nel(1)
                        valid = validYZ & (x + dxOff >= 1) & (x + dxOff <= obj.nel(1));

                        xN = x + dxOff(valid);
                        yN = y + dyOff(valid);
                        zN = z + dzOff(valid);
                        w  = wOff(valid);
                        w  = w ./ sum(w);

                        idx = mapElements(x, y, z);
                        cols = mapElements(sub2ind(obj.nel, xN, yN, zN));
                        nPositive = numel(cols);

                        range = k:(k + nPositive - 1);
                        iRow(range) = idx;
                        iCol(range) = cols;
                        vals(range) = w;
                        k = k + nPositive;
                    end
                end
            end

            iRow = iRow(1:k-1);
            iCol = iCol(1:k-1);
            vals = vals(1:k-1);

            obj.H = sparse(iRow, iCol, vals, obj.nElements, obj.nElements);
        end

        function initialize_helmholtz_from_mesh(obj, nodes, elements, r)
            % Initialize FE-based Helmholtz filter operators.
            %
            % nodes, elements describe a 2D QUAD4 mesh.
            % r is the physical Helmholtz length scale.

            obj.rHelm = r;
            [obj.Kf, obj.Bf, obj.Pf] = build_helmholtz_filter_Q4(nodes, elements, r);

            % Cache factorization for repeated forward/adjoint solves
            try
                obj.KfFactor = decomposition(obj.Kf, 'chol', 'upper');
            catch
                obj.KfFactor = decomposition(obj.Kf);
            end
        end


        function initialize_density(obj, dInit)
            obj.d(:) = dInit;
        end

        function set_initial_density_box(obj, coord, tol, val)
            dist = obj.coord - coord;
            isInside = all(abs(dist) < tol, 2);
            obj.d(isInside) = val;
        end

        function set_density_box(obj, coord, tol, val)
            dist = obj.coord - coord;
            isInside = all(abs(dist) < tol, 2);
            obj.d(isInside) = val;
            obj.freeElements(isInside) = 0;
        end

        function set_initial_density_sphere(obj, coord, tol, val)
            dist = obj.coord - coord;
            isInside = sum(dist.^2, 2) < tol^2;
            obj.d(isInside) = val;
        end

        function set_density_sphere(obj, coord, tol, val)
            dist = obj.coord - coord;
            isInside = sum(dist.^2, 2) < tol^2;
            obj.d(isInside) = val;
            obj.freeElements(isInside) = 0;
        end

        function rhoF = apply_filter(obj, rho)
            % Generic forward filter.

            if ~obj.useFilter || strcmpi(obj.filterType, 'none')
                rhoF = rho;
                return;
            end

            switch lower(obj.filterType)
                case 'conv'
                    rhoF = obj.H * rho;

                case 'helmholtz'
                    if isempty(obj.Kf) || isempty(obj.Bf) || isempty(obj.Pf)
                        error(['Helmholtz operators are empty. ', ...
                               'Call initialize_helmholtz_from_mesh(nodes, elements, r) first.']);
                    end
                    if isempty(obj.KfFactor)
                        q = obj.Kf \ (obj.Bf * rho);
                    else
                        q = obj.KfFactor \ (obj.Bf * rho);
                    end
                    rhoF = obj.Pf * q;

                otherwise
                    error('Unknown filterType: %s', obj.filterType);
            end
        end

        function grad = apply_filter_transpose(obj, gradF)
            % Generic adjoint/transpose filter.

            if ~obj.useFilter || strcmpi(obj.filterType, 'none')
                grad = gradF;
                return;
            end

            switch lower(obj.filterType)
                case 'conv'
                    grad = obj.H.' * gradF;

                case 'helmholtz'
                    if isempty(obj.Kf) || isempty(obj.Bf) || isempty(obj.Pf)
                        error(['Helmholtz operators are empty. ', ...
                               'Call initialize_helmholtz_from_mesh(nodes, elements, r) first.']);
                    end
                    rhs = obj.Pf.' * gradF;
                    if isempty(obj.KfFactor)
                        adj = obj.Kf \ rhs;
                    else
                        adj = obj.KfFactor \ rhs;
                    end
                    grad = obj.Bf.' * adj;

                otherwise
                    error('Unknown filterType: %s', obj.filterType);
            end
        end

        function filter(obj)
            % Legacy-compatible wrapper
            obj.d_filt = obj.apply_filter(obj.d);
        end

        function filtSens = filter_sensitivity(obj, sens)
            % dJ/dd = Filter^T * (dJ/dd_proj .* dd_proj/dd_filt)

            sens = sens .* obj.project_derivative(obj.d_filt, obj.eta);
            filtSens = obj.apply_filter_transpose(sens);
        end

        function rhoP = project_field(obj, rhoF, etaLoc)
            % Tanh threshold projection.

            if ~obj.useProjection
                rhoP = rhoF;
                return;
            end

            if strcmpi(obj.filterType, 'helmholtz')
                if etaLoc <= 0 || etaLoc >= 1
                    error(['For Helmholtz filter, thresholds eta = 0 or eta = 1 ', ...
                           'should be avoided.']);
                end
            end

            denom = tanh(obj.beta * etaLoc) + tanh(obj.beta * (1 - etaLoc));
            rhoP = (tanh(obj.beta * etaLoc) + tanh(obj.beta * (rhoF - etaLoc))) / denom;
        end

        function dproj = project_derivative(obj, rhoF, etaLoc)
            % Derivative of tanh projection wrt filtered density.

            if ~obj.useProjection
                dproj = ones(size(rhoF));
                return;
            end

            if strcmpi(obj.filterType, 'helmholtz')
                if etaLoc <= 0 || etaLoc >= 1
                    error(['For Helmholtz filter, thresholds eta = 0 or eta = 1 ', ...
                           'should be avoided.']);
                end
            end

            denom = tanh(obj.beta * etaLoc) + tanh(obj.beta * (1 - etaLoc));
            dproj = obj.beta * (1 - tanh(obj.beta * (rhoF - etaLoc)).^2) / denom;
        end

        function projection(obj)
            % Legacy single-threshold projection wrapper.
            obj.d_proj = obj.project_field(obj.d_filt, obj.eta);
        end

        function projSens = projection_sensivitity(obj)
            % Legacy single-threshold projection derivative wrapper.
            projSens = obj.project_derivative(obj.d_filt, obj.eta);
        end

        function rhoS = simp_field(obj, rhoP)
            rhoS = obj.dMinSimp + (1 - obj.dMinSimp) * rhoP .^ obj.p;
        end

        function dsimp = simp_derivative(obj, rhoP)
            dsimp = obj.p * (1 - obj.dMinSimp) * rhoP .^ (obj.p - 1);
        end

        function simp(obj)
            % Legacy SIMP wrapper.
            obj.d_simp = obj.simp_field(obj.d_proj);
        end

        function simpSens = simp_sensitivity(obj)
            % Legacy SIMP derivative wrapper.
            simpSens = obj.simp_derivative(obj.d_proj);
        end

        function Mnd = grey_indicator(obj, rhoBar)
            % Grey indicator for a projected/physical density field.
            % Mnd = mean(4*rhoBar*(1-rhoBar))*100 [%%].

            if nargin < 2
                rhoBar = obj.d_proj;
            end

            Mnd = mean(4 .* rhoBar(:) .* (1 - rhoBar(:))) * 100;
        end

        function rhoR = ramp_field(obj, rhoP)
            rhoR = obj.dMinRamp + (1 - obj.dMinRamp) * rhoP ./ ...
                (1 + obj.q * (1 - rhoP));
        end

        function dramp = ramp_derivative(obj, rhoP)
            dramp = (1 - obj.dMinRamp) * (1 + obj.q) ./ ...
                (1 + obj.q * (1 - rhoP)).^2;
        end

        function ramp(obj)
            obj.d_ramp = obj.ramp_field(obj.d_proj);
        end

        function rampSens = ramp_sensitivity(obj)
            rampSens = obj.ramp_derivative(obj.d_proj);
        end
    end
end
