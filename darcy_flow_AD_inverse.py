from hippylib import *
from mshr import *
import dolfin as dl; dl.set_log_level(40)
import ufl
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

# Compute MAP estimate using gradients
from scipy.optimize import minimize, Bounds

# Dimensions of the mesh extremeties
L = 1.0
W = 1.00

class SpaceTimePointwiseStateObservation(Misfit):
    '''Creates a class that represents observations in time and space.

    More information regarding the base class:
        https://hippylib.readthedocs.io/en/2.0.0/_modules/hippylib/modeling/misfit.html

    Inputs:
        observation_times - Array of times to make observations at
        targets - Array of spatial coordinates representing observation points
        data - Input time dependent vector representing data
        noise_variance - Measurement noise
    '''

    def __init__(self, Vh, observation_times, targets, data=None, noise_variance=None):
        self.Vh = Vh
        self.observation_times = observation_times

        # hippylib pointwise observation construction
        self.B = assemblePointwiseObservation(self.Vh, targets)
        self.ntargets = targets

        if data is None:
            self.data = TimeDependentVector(observation_times)
            self.data.initialize(self.B, 0)
        else:
            self.data = data

        # TODO: Currently noise variance is assumed to be a scalar
        self.noise_variance = noise_variance

        # Temporary vectors to store retrieved state, observations, and data
        self.u_snapshot = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.data_snapshot = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)
        self.B.init_vector(self.Bu_snapshot, 0)
        self.B.init_vector(self.data_snapshot, 0)

    def observe(self, x, obs):
        ''' Store observations given time-dependent state into output obs '''
        obs.zero()

        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            obs.store(self.Bu_snapshot, t)

    def cost(self, x):
        ''' Compute misfit cost by summing over all observations in time and space '''
        c = 0
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            self.data.retrieve(self.data_snapshot, t)
            self.Bu_snapshot.axpy(-1., self.data_snapshot)
            c += self.Bu_snapshot.inner(self.Bu_snapshot)

        return c/(2.*self.noise_variance)

    def grad(self, i, x, out):
        ''' Compute the gradient of the cost function with respecto to i = {STATE, PARAMETER} '''
        out.zero()
        if i == STATE:
            # Gradient w.r.t state is simply B^T(Bu - d)
            for t in self.observation_times:
                x[STATE].retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.data.retrieve(self.data_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.data_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot)
                out.store(self.u_snapshot, t)
        else:
            # Gradient w.r.t parameter is zero
            pass

    def grad_reduced(self, i, x, out, phi):
        ''' Compute the gradient of the ROM cost function with respecto to i = {STATE, PARAMETER} '''
        out.zero()

        u_r_snapshot = dl.Vector()
        phi.init_vector(u_r_snapshot, 1)
        B_phi = dl.PETScMatrix(dl.as_backend_type(self.B).mat().matMult(phi.mat()))
        
        if i == STATE:
            # Gradient w.r.t state is simply B^T(Bu - d)
            for t in self.observation_times:
                x.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.data.retrieve(self.data_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.data_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                B_phi.transpmult(self.Bu_snapshot, u_r_snapshot)
                out.store(u_r_snapshot, t)
        else:
            # Gradient w.r.t parameter is zero
            pass

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass


    def apply_ij(self, i, j, direction, out):
        ''' Compute second variation of the cost function in the given direction '''
        out.zero()
        if i == STATE and j == STATE:
            for t in self.observation_times:
                direction.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot)
                out.store(self.u_snapshot, t)
        else:
            # Second variations involving parameters is zero
            pass

class TimeDependentAdvectionDiffusion:
    def __init__(self, mesh, Vh, prior, misfit, simulation_times, velocity, u_0, \
            observation_times, parameter_basis, gls_stab=False, debug=False):
        '''Initialize a time-dependent advection diffusion problem

        Inputs:
            mesh - mesh generated using mshr
            Vh - tuple of three fenics function spaces for state, parameter, and adjoint variables
            prior - hippylib prior for regularization of the inverse problem
            misfit - misfit class which models observations with the ability to evaluate gradients
            simulation_times - array describing simulation time steps
            velocity - velocity function for advection-diffusion
            u_0 - initial condition of concentrate
            gls_stab - Set true to turn on Galerkin Least-Squares stabilization. Currently unsupported
            debug - Turn on debug mode with verbose reporting and plotting
        '''
        self.debug = debug

        # Set member variables describing the problem
        self.mesh = mesh
        self.Vh = Vh
        self.ndofs = Vh[STATE].dim()
        self.prior = prior
        self.misfit = misfit
        self.u_0 = u_0

        # Assume constant timestepping
        self.simulation_times = simulation_times
        self.observation_times = observation_times
        self.dt = simulation_times[1] - simulation_times[0]
        self.B_T = self.misfit.B.array().T 

        # Trial and Test functions for the weak forms
        u_trial = dl.TrialFunction(Vh[STATE])
        p_trial = dl.TrialFunction(Vh[ADJOINT])
        u_test = dl.TestFunction(Vh[STATE])
        p_test = dl.TestFunction(Vh[ADJOINT])

        # Functions to be populated for time stepping
        self.u_old = dl.Function(Vh[STATE])
        self.p_old = dl.Function(Vh[ADJOINT])
        self.kappa = dl.Function(Vh[PARAMETER])
        self.approx_kappa = dl.Function(Vh[PARAMETER])

        dt_expr = dl.Constant(self.dt)

        # Describe sources TODO Make this programmatic
        N_source_grid_size = 4
        source = None
        s_decay = 1000
        s_interval = 0.25
        bottom_left_x = L/2.0 - (N_source_grid_size-1.0) / 2.0 * s_interval
        bottom_left_y = W/2.0 - (N_source_grid_size-1.0) / 2.0 * s_interval

        for i in range(N_source_grid_size):
            for j in range(N_source_grid_size):
                s_x = bottom_left_x + s_interval * i
                s_y = bottom_left_y + s_interval * j

                source_point = dl.Expression('min(0.5, exp(-s*(pow(x[0]-s_x, 2) + pow(x[1]-s_y, 2))))', \
                        element=Vh[STATE].ufl_element(), s=s_decay, s_x=s_x, s_y=s_y)
                if source is None:
                    source = source_point
                else:
                    source += source_point

        ############################################################################################
        # Galerkin Least Squares stabilization terms TODO: Currently unsupported
        r_trial = u_trial + dt_expr * \
            (-ufl.div(self.kappa*ufl.grad(u_trial)) + ufl.inner(velocity, ufl.grad(u_trial)))
        r_test = u_test + dt_expr*(-ufl.div(self.kappa*ufl.grad(u_test)) + ufl.inner(velocity, ufl.grad(u_test)))

        h = dl.CellDiameter(mesh)
        vnorm = ufl.sqrt(ufl.inner(velocity, velocity))
        if gls_stab:
            tau = ufl.min_value((h*h)/(dl.Constant(2.)*self.kappa), h/vnorm)
        else:
            tau = dl.Constant(0.)

        ############################################################################################

        # Mass matrix variational forms and their assembled matrices
        M_varf = ufl.inner(u_trial, u_test)*ufl.dx
        self.M = dl.assemble(M_varf)

        M_stab_varf = ufl.inner(u_trial, u_test + tau * r_test) * ufl.dx
        self.M_stab = dl.assemble(M_stab_varf)

        Mt_stab_varf = ufl.inner(u_trial + tau * r_trial, u_test) * ufl.dx
        self.Mt_stab = dl.assemble(Mt_stab_varf)

        # Variational form for time-stepping
        N_varf = (ufl.inner(self.kappa * ufl.grad(u_trial), ufl.grad(u_test)) \
                + ufl.inner(velocity, ufl.grad(u_trial)) * u_test) * ufl.dx
        Nt_varf = (ufl.inner(self.kappa * ufl.grad(p_test), ufl.grad(p_trial)) \
                + ufl.inner(velocity, ufl.grad(p_test)) * p_trial) * ufl.dx

        stab_varf = tau*ufl.inner(r_trial, r_test) * ufl.dx

        # LHS variational form to be solved at each time step
        self.L_varf = M_varf + dt_expr * N_varf + stab_varf
        self.S_varf = dt_expr * ufl.inner(source, u_test) * ufl.dx
        self.L_rhs_varf = ufl.inner(self.u_old, u_test) * ufl.dx
        self.Lt_varf = M_varf + dt_expr * Nt_varf + stab_varf
        self.Lt_rhs_varf = ufl.inner(self.p_old, p_test) * ufl.dx

        # Part of model public API for hippylib
        self.gauss_newton_approx = False

        # Setup variational forms for gradient evaluation
        self.solved_u = dl.Function(Vh[STATE])
        self.solved_p = dl.Function(Vh[ADJOINT])
        self.solved_u_tilde = dl.Function(Vh[STATE])
        self.solved_p_tilde = dl.Function(Vh[ADJOINT])
        self.k_hat = dl.TestFunction(Vh[PARAMETER])
        self.grad_form = ufl.inner(self.k_hat * ufl.grad(self.solved_u), ufl.grad(self.solved_p)) * ufl.dx

        # Setup required storage for Hessian evaluation
        # Refer this guide for a reference: http://g2s3.com/labs/notebooks/Poisson_INCG.html
        self.u_s = TimeDependentVector(self.simulation_times) # Solved solution values
        self.u_s.initialize(self.M, 0)
        self.p_s = TimeDependentVector(self.simulation_times) # Solved adjoint values
        self.p_s.initialize(self.M, 0)
        self.k_first_var = dl.Function(Vh[PARAMETER]) # First variation direction of parameter
        self.C_form = ufl.inner(self.k_first_var * ufl.grad(self.solved_u), ufl.grad(u_test)) * ufl.dx
        self.W_um_form = ufl.inner(self.k_first_var * ufl.grad(p_test), ufl.grad(self.solved_p)) * ufl.dx

        # Setup required objects for solving reduced order models
        phi = None
        self.L_r = None
        self.M_r = None
        self.S_r = None
        self.u_tildes = TimeDependentVector(self.simulation_times)
        self.u_tildes.initialize(self.M, 0)
        self.p_tildes = TimeDependentVector(self.simulation_times)
        self.p_tildes.initialize(self.M, 0)
        self.L_u_tildes = TimeDependentVector(self.simulation_times)
        self.L_u_tildes.initialize(self.M, 0)
        self.L_p_tildes = TimeDependentVector(self.simulation_times)
        self.L_p_tildes.initialize(self.M, 0)
        # TODO, have L_phi and M_phi here for PG projection speed-up once affine decomposition is done

        # Setup required abjects for a posteriori error estimation
        self.approximate_residuals = TimeDependentVector(self.simulation_times)
        self.approximate_residuals.initialize(self.M, 0)
        self.true_qoi_errors = np.zeros((len(self.observation_times), len(self.misfit.ntargets)))
        self.qoi_bounds = np.zeros((len(self.observation_times), len(self.misfit.ntargets)))
        self.approx_qoi_bounds = np.zeros((len(self.observation_times), len(self.misfit.ntargets)))

        # Karhunen-Loeve expansion of parameter. Accumulate the affine diffusion component of the operator
        self.KL_dim = parameter_basis.shape[1]
        self.parameter_basis = parameter_basis
        self.parameter_proj = np.dot(self.parameter_basis, self.parameter_basis.T)
        self.KL_coefs = []
        self.KL_varfs = []
        self.KL_diff_varf = None 
        for ii in range(self.KL_dim):
            KL_coef = dl.Constant(1.0)
            self.KL_coefs.append(KL_coef)
            KL_eig_f = dl.Function(Vh[PARAMETER])
            KL_eig_f.vector().set_local(parameter_basis[:, ii])
            varf = ufl.inner(KL_eig_f * ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
            self.KL_varfs.append(varf)
            diff_varf = self.KL_coefs[ii] * varf
            if self.KL_diff_varf is None:
                self.KL_diff_varf = diff_varf
            else:
                self.KL_diff_varf += diff_varf

        # LHS variational form to be solved at each time step using affine decomposition
        vel_varf = ufl.inner(velocity, ufl.grad(u_trial)) * u_test * ufl.dx
        self.affine_L_varf = M_varf + dt_expr * self.KL_diff_varf + dt_expr * vel_varf

    def set_reduced_basis(self, phi):
        dofs, n_basis = phi.shape
        self.phi = dl.PETScMatrix(PETSc.Mat().createDense([dofs, n_basis], array=phi))

    def generate_vector(self, component="ALL"):
        '''Generates an appropriately initialized PETSc vector for appropriate variables'''
        if component == "ALL":
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            m = dl.Vector()
            self.prior.init_vector(m, 0)
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return [u, m, p]
        elif component == STATE:
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            return u
        elif component == PARAMETER:
            m = dl.Vector()
            self.prior.init_vector(m, 0)
            return m
        elif component == ADJOINT:
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return p
        elif component == "REDUCED_STATE":
            u_r = TimeDependentVector(self.simulation_times)
            u_r.initialize(self.phi, 1)
            return u_r
        elif component == "REDUCED":
            u_r = TimeDependentVector(self.simulation_times)
            u_r.initialize(self.phi, 1)
            m = dl.Vector()
            self.prior.init_vector(m, 0)
            p_r = TimeDependentVector(self.simulation_times)
            p_r.initialize(self.phi, 1)
            return [u_r, m, p_r]
        else:
            raise

    def init_parameter(self, m):
        '''Initialize parameter to be compatible with the prior'''
        self.prior.init_vector(m, 0)

    def cost(self, x):
        '''Evaluate the cost functional to be optimized for the inverse problem'''
        Rdx = dl.Vector()
        self.prior.init_vector(Rdx, 0)
        dx = x[PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, Rdx)
        reg = .5*Rdx.inner(dx)

        misfit = self.misfit.cost(x)

        return [reg+misfit, reg, misfit]

    def solveFwd(self, out, x):
        '''Perform implicit time-stepping and solve the forward problem '''

        out.zero()
        self.u_s.zero()

        # Set initial condition
        self.u_old.assign(self.u_0)
        out.store(self.u_old.vector(), 0.)
        self.u_s.store(self.u_old.vector(), 0.)

        # Assemble LHS dependent on parameters
        self.kappa.vector().set_local(x[PARAMETER])
        #  self.approx_kappa.vector().set_local(np.dot(self.parameter_proj, self.kappa.vector()[:]))
        #  self.kappa.assign(self.approx_kappa)

        u = dl.Function(self.Vh[STATE])

        for t in self.simulation_times[1::]:
            rhs = self.L_rhs_varf + self.S_varf
            dl.solve(self.L_varf == rhs, u)
            out.store(u.vector(), t)
            self.u_s.store(u.vector(), t) #TODO Fix duplicate storage
            self.u_old.assign(u)

    def solveReducedFwd(self, out, x):
        '''Perform implicit time-stepping and solve the forward problem in the 
        reduced subpsace determined by the basis phi. Uses a Petrov-Galerkin projection'''
        out.zero()

        # Set reduced initial condition
        u_r = dl.Vector() # Coordinates of the reduced space
        self.phi.init_vector(u_r, 1) # u = Phi * u_r
        self.phi.transpmult(self.u_0.vector(), u_r)
        out.store(u_r, 0.)

        # Store approximate solutions using ROMs in the original state space
        self.u_tildes.zero()
        # TODO : Project and unproject w/ phi
        self.u_tildes.store(self.u_0.vector(), 0.)
        u_tilde = dl.Vector()
        self.M.init_vector(u_tilde, 0)

        self.L_u_tildes.zero()
        L_u_tilde = dl.Vector()
        self.M.init_vector(L_u_tilde, 0)

        # Right-hand side of reduced system of equations
        rhs = dl.Vector()
        self.phi.init_vector(rhs, 1)

        # TODO replace this with affine decomposition to avoid assembly when parameters change
        self.kappa.vector().set_local(x[PARAMETER])
        #  self.approx_kappa.vector().set_local(np.dot(self.parameter_proj, self.kappa.vector()[:]))
        #  self.kappa.assign(self.approx_kappa)
        L = dl.as_backend_type(dl.assemble(self.L_varf)).mat()

        # KL affine decomposition operator assembly. Can be precomputed
        #  KL_coefs = np.dot(self.parameter_basis.T, self.kappa.vector()[:])
        #  for ii in range(self.KL_dim):
            #  self.KL_coefs[ii].assign(KL_coefs[ii])
        #  L = dl.as_backend_type(dl.assemble(self.affine_L_varf)).mat()


        Psi = L.matMult(self.phi.mat()) # Petrov-Galerkin projection
        #  Psi = self.phi.mat() #Galerkin projection

        # Left-hand size of reduced system of equations
        self.L_r = dl.PETScMatrix(Psi.transposeMatMult(Psi))

        # Reduced source term # TODO: Could be optimized and precomputed w/ affine decomposition
        S = dl.assemble(self.S_varf)
        self.S_r = dl.Vector()
        Psi_p = dl.PETScMatrix(Psi)
        Psi_p.transpmult(S, self.S_r)

        # Reduced mass matrix #TODO: Precompute w/ affine decomposition
        M_phi = dl.as_backend_type(self.M).mat().matMult(self.phi.mat())
        self.M_r = dl.PETScMatrix(Psi.transposeMatMult(M_phi))


        # Dual-weighted residual stuff
        M_u_tilde_prev = dl.Vector()
        self.M.init_vector(M_u_tilde_prev, 0)
        B_phi = np.dot(self.misfit.B.array(), self.phi.mat().getDenseArray())
        #  lam = np.linalg.solve(self.L_r.array().T, -B_phi.T) #TODO: incorrect lambda


        for t_idx, t in enumerate(self.simulation_times[1::]):
            self.M.mult(u_tilde, M_u_tilde_prev)
            self.M_r.mult(u_r, rhs)
            rhs.axpy(1., self.S_r)
            dl.solve(self.L_r, u_r, rhs, "cg")
            out.store(u_r, t)
            self.phi.mult(u_r, u_tilde)
            self.u_tildes.store(u_tilde, t)
            dl.PETScMatrix(L).mult(u_tilde, L_u_tilde)
            self.L_u_tildes.store(L_u_tilde, t)
            residual_fom = L_u_tilde - S - M_u_tilde_prev
            self.approximate_residuals.store(residual_fom, t)

            if t in self.observation_times:
                obs_idx = np.searchsorted(self.observation_times, t)
                u = dl.Vector()
                self.M.init_vector(u, 0)
                self.u_s.retrieve(u, t)
                true_e = np.dot(self.misfit.B.array(), u[:]) - np.dot(self.misfit.B.array(), u_tilde[:])
                self.true_qoi_errors[obs_idx, :] = true_e

    def solveAdj(self, out, x):
        '''Solve adjoint problem backwards in time and store in out '''
        out.zero()
        self.p_s.zero()
        self.kappa.vector().set_local(x[PARAMETER])
        #  self.approx_kappa.vector().set_local(np.dot(self.parameter_proj, self.kappa.vector()[:]))
        #  self.kappa.assign(self.approx_kappa)

        grad_state = TimeDependentVector(self.simulation_times)
        grad_state.initialize(self.M, 0)
        self.misfit.grad(STATE, x, grad_state)
        
        p = dl.Vector()
        self.M.init_vector(p, 0)
        self.p_old.vector().set_local(p)

        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)

        grad_state_snap = dl.Vector()
        self.M.init_vector(grad_state_snap, 0)

        residual_w_reduced = dl.Vector()
        self.M.init_vector(residual_w_reduced, 0)
        print(f"Max kappa: {np.max(self.kappa.vector()[:])}, min kappa: {np.min(self.kappa.vector()[:])}")

        num_targets = len(self.misfit.ntargets)
        num_obs_times = len(self.observation_times)
        num_sim_times = len(self.simulation_times)
        lam = np.zeros((num_sim_times, self.ndofs, num_targets))
        L_T = dl.assemble(self.Lt_varf).array()
        M_T = self.Mt_stab.array()
        self.qoi_bounds[:] = 0.0
        self.approx_qoi_bounds[:] = 0.0
        lam_old = np.linalg.solve(L_T, self.B_T)
        lam[-1, :, :] = lam_old

        for t_idx, t in enumerate(self.simulation_times[::-1]):
            Lt, rhs = dl.assemble_system(self.Lt_varf, self.Lt_rhs_varf)

            #####################################################################
            # Dual-weighted residual computation
            # TODO: Solve adjoint in a reduced space
            self.approximate_residuals.retrieve(residual_w_reduced, t)

            obs_idx = np.searchsorted(self.observation_times, t)
            if t_idx > 0:
                lam_old = np.linalg.solve(L_T, np.dot(M_T, lam_old))
                lam[-(1+t_idx), :, :] = lam_old

            for j in range(obs_idx, num_obs_times):
                observation_time = self.observation_times[j]
                timesteps_since_observation = int(np.rint((observation_time - t)/self.dt))
                lam_idx = -(1 + timesteps_since_observation)
                self.qoi_bounds[j, :] += np.dot(-lam[lam_idx, :, :].T, residual_w_reduced[:])
                self.approx_qoi_bounds[j, :] += np.dot(np.abs(lam[lam_idx, :, :].T), np.abs(residual_w_reduced[:]))
            #####################################################################

            grad_state.retrieve(grad_state_snap, t)
            rhs.axpy(-0.1, grad_state_snap)

            dl.solve(Lt, p, rhs)
            self.p_old.vector().set_local(p)
            out.store(p, t)
            self.p_s.store(p, t) #TODO Fix duplicate storage

        #  if not np.allclose(self.true_qoi_errors, self.qoi_bounds):
        if not np.all(self.approx_qoi_bounds > self.true_qoi_errors):
            import pdb; pdb.set_trace()
        return True

    def solveReducedAdj(self, out, x):
        '''Perform implicit time-stepping and solve the adjoint problem in the 
        reduced subpsace determined by the basis phi. Uses a Petrov-Galerkin projection'''
        out.zero()
        self.p_tildes.zero()
        p_tilde = dl.Vector()
        self.M.init_vector(p_tilde, 0)

        self.L_p_tildes.zero()
        L_p_tilde = dl.Vector()
        self.M.init_vector(L_p_tilde, 0)
        L = dl.as_backend_type(dl.assemble(self.L_varf))

        grad_state = TimeDependentVector(self.simulation_times)
        grad_state.initialize(self.phi, 1)
        grad_state_snap = dl.Vector() 
        self.phi.init_vector(grad_state_snap, 1)
        self.misfit.grad_reduced(STATE, self.u_tildes, grad_state, self.phi)

        # Set reduced initial condition
        p_r = dl.Vector()
        self.phi.init_vector(p_r, 1) # p_tilde = Phi * p_r

        rhs = dl.Vector()
        self.phi.init_vector(rhs, 1)

        L_r_t = dl.PETScMatrix(self.L_r.mat().transpose())

        for t in self.simulation_times[1::]:
            self.M_r.transpmult(p_r, rhs)
            grad_state.retrieve(grad_state_snap, t)
            rhs.axpy(-1., grad_state_snap)
            dl.solve(L_r_t, p_r, rhs, "cg")
            out.store(p_r, t)
            self.phi.mult(p_r, p_tilde)
            self.p_tildes.store(p_tilde, t)
            L.mult(p_tilde, L_p_tilde)
            self.L_p_tildes.store(L_p_tilde, t)

    def evalGradientParameter(self, x, mg, misfit_only=False, use_ROM=False):
        '''Evaluate gradient with respect to parameters'''
        self.prior.init_vector(mg, 1)
        mg.zero()
        if misfit_only == False:
            dm = x[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)

        if use_ROM:
            u = dl.Vector()
            self.M.init_vector(u, 0)
            p = dl.Vector()
            self.M.init_vector(p, 0)
            for t in simulation_times[1::]:
                self.u_tildes.retrieve(self.solved_u.vector(), t)
                self.L_p_tildes.retrieve(self.solved_p.vector(), t)
                mg.axpy(1., dl.assemble(self.grad_form))
                self.L_u_tildes.retrieve(self.solved_u.vector(), t)
                self.p_tildes.retrieve(self.solved_p.vector(), t)
                mg.axpy(1., dl.assemble(self.grad_form))
        else:
            for t in simulation_times[1::]:
                x[STATE].retrieve(self.solved_u.vector(), t)
                x[ADJOINT].retrieve(self.solved_p.vector(), t)
                mg.axpy(1., dl.assemble(self.grad_form))

        g = dl.Vector()
        self.M.init_vector(g,1)
        
        try:
            self.prior.Msolver.solve(g,mg)
        except RuntimeError:
            import pdb; pdb.set_trace()
        
        grad_norm = g.inner(mg)

        return grad_norm

    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        '''Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated. Nothing to do as internally solved values are saved.
        '''
        self.gauss_newton_approx = gauss_newton_approx
        return

    def applyC(self, first_variation, out):
        out.zero()
        self.k_first_var.vector().set_local(first_variation)

        for t in self.simulation_times[1:]:
            self.u_s.retrieve(self.solved_u.vector(), t)
            out.store(dl.assemble(self.C_form), t)

    def solveFwdIncremental(self, sol, rhs):
        '''Solved the incremental forward problem obtained by taking the variation w.r.t.
        to adjoint variables in the meta-Lagrangian
        '''
        sol.zero()

        # Set initial condition
        self.u_old.vector().zero() # Zero initial condition for incremental forward
        sol.store(self.u_old.vector(), 0.)

        u = dl.Vector()
        self.M.init_vector(u, 0)

        C_rhs_n = dl.Vector()
        self.M.init_vector(C_rhs_n, 0)

        for t in self.simulation_times[1::]:
            L_incr, rhs_incr = dl.assemble_system(self.L_varf, self.L_rhs_varf)
            rhs.retrieve(C_rhs_n, t)
            rhs_incr.axpy(-0.1, C_rhs_n)

            dl.solve(L_incr, u, rhs_incr)

            sol.store(u, t)
            self.u_old.vector().set_local(u)
    
    def applyWuu(self, du, out):
        '''Compute second variation of the misfit w.r.t state variables'''
        out.zero()
        self.misfit.apply_ij(STATE, STATE, du, out)
        for vec in out.data:
            vec *= -1.0

    def applyWum(self, first_variation, out):
        out.zero()
        self.k_first_var.vector().set_local(first_variation)

        for t in self.simulation_times[::-1]:
            self.p_s.retrieve(self.solved_p.vector(), t)
            out.store(dl.assemble(self.W_um_form), t)

    def solveAdjIncremental(self, sol, rhs):
        sol.zero()

        p = dl.Vector()
        self.M.init_vector(p, 0)
        self.p_old.vector().set_local(p)

        rhs_t = dl.Vector()
        self.M.init_vector(rhs_t, 0)

        for t in self.simulation_times[::-1]:
            Lt_incr, rhs_incr = dl.assemble_system(self.Lt_varf, self.Lt_rhs_varf)
            rhs.retrieve(rhs_t, t)
            rhs_incr.axpy(0.1, rhs_t)

            dl.solve(Lt_incr, p, rhs_incr)
            self.p_old.vector().set_local(p)
            sol.store(p, t)

    def applyCt(self, dp, out):
        out.zero()
        for t in simulation_times[1::]:
            self.u_s.retrieve(self.solved_u.vector(), t)
            dp.retrieve(self.solved_p.vector(), t)
            out.axpy(1., dl.assemble(self.grad_form))

    def applyWmu(self, du, out):
        out.zero()
        for t in simulation_times[1::]:
            self.p_s.retrieve(self.solved_p.vector(), t)
            du.retrieve(self.solved_u.vector(), t)
            out.axpy(-1., dl.assemble(self.grad_form))

    def applyR(self, dm, out):
        self.prior.R.mult(dm, out)

    def applyWmm(self, dm, out):
        out.zero()

    def exportState(self, x, filename, varname):
        ''' TODO: Update these to be consistent with the problem being solved'''
        out_file = dl.XDMFFile(self.Vh[STATE].mesh().mpi_comm(), filename)
        out_file.parameters["functions_share_mesh"] = True
        out_file.parameters["rewrite_function_mesh"] = False
        ufunc = dl.Function(self.Vh[STATE], name=varname)
        t = self.simulation_times[0]
        out_file.write(vector2Function(
            x[PARAMETER], self.Vh[STATE], name=varname), t)
        for t in self.simulation_times[1:]:
            x[STATE].retrieve(ufunc.vector(), t)
            out_file.write(ufunc, t)

def boundary(x, on_boundary):
    '''Returns true if on the boundary'''
    return on_boundary

def q_boundary(x, on_boundary):
    '''Boundary for pressure'''
    return x[0] > 1.0 - dl.DOLFIN_EPS

def left_boundary(x, on_boundary):
    '''In-flow boundary from the left'''
    return x[0] < dl.DOLFIN_EPS

def right_boundary(x, on_boundary):
    '''In-flow boundary from the right'''
    return x[0] > 1.0 - dl.DOLFIN_EPS

def v_boundary(x, on_boundary):
    '''Velocity boundary prescribed except on right-corner'''
    return on_boundary and x[0] < 1.0 - dl.DOLFIN_EPS

def no_slip_boundary(x, on_boundary):
    '''Boundary expect the left and right edges where velocity and concentrate is zero'''
    return on_boundary

def left_slit_boundary(x, on_boundary):
    '''In-flow boundary from the left through a slit'''
    return (x[0] < dl.DOLFIN_EPS and (x[1] > 0.45 and x[1] < 0.55))

def top_bot_boundary(x, on_boundary):
    ''' Top and bottom boundaries '''
    return (x[1] < dl.DOLFIN_EPS and x[1] > 1.0 - dl.DOLFIN_EPS)

def computeVelocityField(mesh, plot_velocity=False):
    '''Solves the steady Stokes flow given the geometry and the in-flow conditions.
    This velocity field is computed from the DFG benchmark
    http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
    '''

    # TODO Pick appropriate spaces for couple Darcy flow and transport
    Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)

    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    # Permeability tensor (assumed to be isotropic)
    K = dl.Function(Wh)
    correlation_length = 0.15
    prior_std_dev = 2.0
    delta = 1.0/np.sqrt(correlation_length * prior_std_dev)
    gamma = delta * correlation_length * correlation_length
    prior = BiLaplacianPrior(Wh, gamma, delta, robin_bc=True)

    prior.mean = dl.interpolate(dl.Constant(1.0), Wh).vector()
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    true_kappa = dl.Vector()
    prior.init_vector(true_kappa, 0)
    parRandom.normal(1., noise)
    prior.sample(noise, true_kappa)
    sampled_values = np.exp(true_kappa[:]); true_kappa.set_local(sampled_values)
    K.vector().set_local(true_kappa)

    if plot_velocity:
        nb.plot(K, mytitle="Permeability field"); plt.show()


    # Viscosity (assumed to be scalar)
    mu = dl.Constant('1.0')

    # Boundary conditions

    # Define function G such that G \cdot n = g (where n is the outward facing normal)
    class BoundarySource(dl.UserExpression):
       def __init__(self, mesh, **kwargs):
           super().__init__(**kwargs)
           self.mesh = mesh
           self.max_velocity = 0.1
       def eval_cell(self, values, x, ufc_cell):
           cell = dl.Cell(self.mesh, ufc_cell.index)
           n = cell.normal(ufc_cell.local_facet)
           # Parabolic in-flow flux
           g = -4.0 * self.max_velocity * x[1] * (W - x[1])/ (W * W)
           values[0] = (x[0] < dl.DOLFIN_EPS) * g*n[0]
           values[1] = 0.0
       def value_shape(self):
           return (2,)

    G = BoundarySource(mesh)

    bc1 = dl.DirichletBC(XW.sub(0), G, v_boundary)
    bcs = [bc1]

    vq = dl.Function(XW)
    (v,q) = ufl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)

    # Variational form of Darcy flow #TODO Verify this
    # Assumes zero pressure on boundary
    F = (ufl.dot(v, v_test) - K / mu * ufl.div(v_test) * q + ufl.div(v) * q_test) * ufl.dx

    dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                         {"relative_tolerance":1e-4, "maximum_iterations":100}})

    vh = dl.project(v,Xh)
    vh_norm = dl.project(dl.sqrt(dl.inner(v,v)), Wh)
    qh = dl.project(q,Wh)

    if plot_velocity:
        nb.plot(vh, subplot_loc=211, mytitle="Velocity")
        nb.plot(qh, subplot_loc=212,mytitle="Pressure")
        plt.show()

    return v

if __name__ == "__main__":
    try:
        dl.set_log_active(False)
    except:
        pass

    # Turn on debug mode to plot all quantities and print debug values
    debug = True

    #np.random.seed(123)
    sep = "\n"+"#"*80+"\n"

    # Discretization parameters
    N_size = 40

    # Define domain
    geometry = Rectangle(dl.Point(0.0, 0.0), dl.Point(L, W)) 

    # Build mesh
    mesh = generate_mesh(geometry, N_size)
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())


    # Function space for state, adjoint, and parameter variables are chosen to be the same
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

    velocity = computeVelocityField(mesh, debug)

    if rank == 0:
        print(sep, "Set up the mesh and finite element spaces.\n",
              "Compute wind velocity", sep)

    ndofs = Vh.dim()
    if rank == 0:
        print("Number of dofs: {0}".format(ndofs))

    if rank == 0:
        print(sep, "Set up Prior Information and model", sep)

    ic_expr = dl.Expression('0.0',element=Vh.ufl_element())

    u_0 = dl.interpolate(ic_expr, Vh)
    if debug:
        nb.plot(u_0, mytitle="Initial condition interpolated")
        plt.show()

    # Gaussian priors in infinite dimensions are created following this guide 
    # http://g2s3.com/labs/notebooks/Gaussian_priors.html and using hippylib. 
    # The prior is parametrized by the correlation length and the pointwise variance
    correlation_length = 0.5
    prior_std_dev = 0.005
    gamma = 1.0/(correlation_length * prior_std_dev)
    delta = gamma/(correlation_length * correlation_length)
    prior = BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)

    # The true diffusivity is drawn from the same distribution as the prior but 
    # with a different mean
    prior.mean = dl.interpolate(dl.Constant(0.002), Vh).vector()
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    true_kappa = dl.Vector()
    true_kappa_f = dl.Function(Vh)
    prior.init_vector(true_kappa, 0)
    parRandom.normal(1., noise)
    prior.sample(noise, true_kappa)
    sampled_values = true_kappa[:]; true_kappa.set_local(sampled_values)

    prior.mean = dl.interpolate(dl.Constant(0.004), Vh).vector()

    # Karhunen-Loeve expansion of parameters
    n_parameter_samples = 200
    parameter_samples = np.zeros((Vh.dim(), n_parameter_samples))
    sample = dl.Vector()
    prior.init_vector(sample, 0)
    for parameter_sample_idx in range(n_parameter_samples):
        parRandom.normal(1., noise)
        prior.sample(noise, sample)
        parameter_samples[:, parameter_sample_idx] = sample[:]
    UU, SS, VV = np.linalg.svd(parameter_samples)

    KL_dim = 10
    parameter_basis = UU[:, :KL_dim]

    # Visualize draws from the prior for debugging purposes
    if debug:
        noise = dl.Vector()
        prior.init_vector(noise, "noise")
            
        sample = dl.Vector()
        prior.init_vector(sample, 0)
            
        ss = []
            
        for i in range(6):
            parRandom.normal(1., noise)
            prior.sample(noise, sample)
            sampled_values = sample[:]; sample.set_local(sampled_values)
            ss.append(vector2Function(sample, Vh))
                
        nb.multi1_plot(ss[0:3], ["sample 1", "sample 2", "sample 3"])
        nb.multi1_plot(ss[3:6], ["sample 4", "sample 5", "sample 6"])
        plt.show()

    if rank == 0:
        print( "Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma, 2))


    # Define simulation and observation times. Observations are made every other time step after t_init
    t_init = 0.
    t_final = 6.
    t_1 = 0.2
    dt = .1
    observation_dt = .2

    simulation_times = np.arange(t_init, t_final+.5*dt, dt)
    #  observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)
    observation_times = simulation_times[round(t_1/dt)::round(observation_dt/dt)]

    # Define observation targets. Chosen randomly on the domain
    ntargets = 20
    targets = np.random.uniform([0.0, 0.0], [L, W], (ntargets, 2))
    if rank == 0:
        print("Number of observation points: {0}".format(targets.shape[0]))

    # Observation misfit defined to be pointwise over space and time
    misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)

    problem = TimeDependentAdvectionDiffusion(
        mesh, [Vh, Vh, Vh], prior, misfit, simulation_times, velocity, u_0, observation_times, parameter_basis, False, debug)
    #  problem.left_inflow.apply(u_0.vector())

    if rank == 0:
        print(sep, "Generate synthetic observation", sep)

    rel_noise = 0.01

    utrue = problem.generate_vector(STATE)
    x = [utrue, true_kappa, None]
    problem.solveFwd(x[STATE], x)
    
    misfit.observe(x, misfit.data)
    MAX = misfit.data.norm("linf", "linf")
    noise_std_dev = rel_noise * MAX

    print(f"Max observed value: {MAX}\n")
    print(f"Standard deviation of noise: {noise_std_dev}\n")
    parRandom.normal_perturb(noise_std_dev, misfit.data)
    misfit.noise_variance = noise_std_dev*noise_std_dev

    if debug:
        nb.show_solution(Vh, u_0.vector(), x[STATE], mytitle="Solution", times=np.linspace(t_init, t_final, 6))
        plt.show()


    ####################################################################
    # Reduced Problem Testing

    # TODO: Use a different prior mean
    prior_mean = prior.mean
    prior.mean = true_kappa

    # Create reduced-space basis sampling from the prior # Replace with model-constrained adaptive sampling
    S_full = None
    for prior_sample in range(10):
        noise = dl.Vector()
        prior.init_vector(noise, "noise")
            
        kappa = dl.Vector()
        prior.init_vector(kappa, 0)
            
        parRandom.normal(1., noise)
        prior.sample(noise, kappa)

        snapshots = problem.generate_vector(STATE)
        x_snapshots = [snapshots, kappa, None]
        problem.solveFwd(x_snapshots[STATE], x_snapshots)

        Y = np.zeros((len(simulation_times), Vh.dim()))
        S = np.zeros((Vh.dim(), len(simulation_times) - 1))
        ii = 0
        for u in snapshots.data:
            weight = dt
            if ii==0 or ii==(len(simulation_times)-1):
                weight /= 2
            Y[ii, :] = np.sqrt(weight) * u[:]
            if ii > 0:
                S[:, ii-1] = u[:] - u_0.vector()[:]
            ii += 1
        if S_full is None:
            S_full = S
        else:
            S_full = np.column_stack((S_full, S))
            
    UU, SS, VV = np.linalg.svd(S_full)

    if debug:
        plt.semilogy(SS); plt.show()
    pod_thresh = 1e-13
    basis_size = np.sum(SS > pod_thresh)
    print(f"Number of singular values larger than {pod_thresh}: {basis_size}")

    #  K = np.dot(Y, Y.T)
    #  e, v = np.linalg.eig(K)

    #  U = np.zeros((basis_size, Vh.dim()))
    #  for i in range(basis_size):
        #  e_i = v[:,i].real
        #  U[i,:] = np.sum(np.dot(np.diag(e_i), Y),0)
    #  basis = U.T
    basis = UU[:, :basis_size]
    print(f"Dimension of reduced basis: {basis.shape[1]}")

    problem.set_reduced_basis(basis)
    prior.mean = prior_mean

    def solve_reduced_w_error_estimate(kappa):
        u_s = problem.generate_vector(STATE)
        p_s = problem.generate_vector(ADJOINT)
        u_FOM = [u_s, kappa, p_s]
        problem.solveFwd(u_FOM[STATE], u_FOM)

        utrue_r = problem.generate_vector("REDUCED_STATE")
        x = [utrue_r, kappa, None]
        problem.solveReducedFwd(x[STATE], x)
        problem.solveAdj(u_FOM[ADJOINT], u_FOM) # Computes full adjoint and estimates bound on error

    solve_reduced_w_error_estimate(true_kappa)

    utrue_r = problem.generate_vector("REDUCED_STATE")
    ptrue_r = problem.generate_vector("REDUCED_STATE")
    x = [utrue_r, true_kappa, ptrue_r]
    problem.solveReducedFwd(x[STATE], x)
    if debug:
        nb.show_solution(Vh, u_0.vector(), problem.u_tildes, mytitle="Solution RB", \
                times=np.linspace(t_init, t_final, 6))
        plt.show()
    problem.solveReducedAdj(x[ADJOINT], x)
    mg = problem.generate_vector(PARAMETER)
    grad_norm_r = problem.evalGradientParameter(x, mg, use_ROM=True)
    print(f"Norm of the reduced gradient {grad_norm_r}")

    if debug:
        h = problem.generate_vector(PARAMETER)
        parRandom.normal(1., h)
        
        u_r = problem.generate_vector("REDUCED_STATE")
        p_r = problem.generate_vector("REDUCED_STATE")
        x_r = [u_r, true_kappa, p_r]
        problem.solveReducedFwd(x[0], x)
        problem.solveReducedAdj(x[2], x)
        x = [problem.u_tildes, true_kappa, problem.p_tildes]
        cx = problem.cost(x)
        
        grad_x = problem.generate_vector(PARAMETER)
        problem.evalGradientParameter(x, grad_x, misfit_only=False, use_ROM=True)
        grad_xh = grad_x.inner( h )
        
        n_eps = 24
        eps = np.power(.5, np.arange(2, n_eps+2))
        err_grad = np.zeros(n_eps)
        
        for i in range(n_eps):
            my_eps = eps[i]
            
            u_r = problem.generate_vector("REDUCED_STATE")
            p_r = problem.generate_vector("REDUCED_STATE")
            k = problem.generate_vector(PARAMETER)
            x_r_plus = [u_r, k, p_r]
            x_r_plus[1].axpy(1., true_kappa)
            x_r_plus[1].axpy(my_eps, h)
            problem.solveReducedFwd(x_r_plus[0], x_r_plus)
            problem.solveReducedAdj(x_r_plus[2], x_r_plus)
            x_plus = [problem.u_tildes, k, problem.p_tildes]
            
            dc = problem.cost(x_plus)[0] - cx[0]
            err_grad[i] = abs(dc/my_eps - grad_xh)

        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check for Reduced Model")
        plt.show()

    def reduced_cost_function(parameter_value):
        '''Cost function for reduced problem in numpy'''
        u_r = problem.generate_vector("REDUCED_STATE")
        p_r = problem.generate_vector("REDUCED_STATE")
        k = problem.generate_vector(PARAMETER)
        k.set_local(parameter_value)
        x_r = [u_r, k, p_r]
        problem.solveReducedFwd(x_r[STATE], x_r)
        problem.solveReducedAdj(x_r[ADJOINT], x_r)
        x = [problem.u_tildes, k, problem.p_tildes]
        return problem.cost(x)[0]

    def reduced_gradient(parameter_value):
        '''Gradient of the cost function for the reduced problem in numpy'''
        u_r = problem.generate_vector("REDUCED_STATE")
        p_r = problem.generate_vector("REDUCED_STATE")
        k = problem.generate_vector(PARAMETER)
        k.set_local(parameter_value)
        x_r = [u_r, k, p_r]
        problem.solveReducedFwd(x_r[STATE], x_r)
        problem.solveReducedAdj(x_r[ADJOINT], x_r)
        grad_x = problem.generate_vector(PARAMETER)
        problem.evalGradientParameter(x_r, grad_x, misfit_only=False, use_ROM=True)
        return grad_x[:]

    solve_reduced_inverse = True

    if solve_reduced_inverse:
        starting_parameter_values = prior.mean[:]
        #  starting_parameter_values = dl.interpolate(dl.Constant(0.003), Vh).vector()[:]
        print(f'Starting cost: {reduced_cost_function(starting_parameter_values)}')
        bounds = Bounds(0.001, 0.006)
        res = minimize(reduced_cost_function, starting_parameter_values, 
                method='L-BFGS-B', 
                jac=reduced_gradient,
                bounds=bounds,
                options={'ftol':1e-8, 'gtol':1e-8, 'maxls':20, 'iprint':11})
        print(f'\nstatus: {res.success}, message: {res.message}, n_it: {res.nit}')
        print(f'Minimum cost: {res.fun:.3F}')

        reconstruction_ROM = problem.generate_vector(PARAMETER)
        reconstruction_ROM.set_local(res.x)
        print(f"Relative reconstruction error with ROM forward: {dl.norm(reconstruction_ROM - true_kappa)/dl.norm(true_kappa)}")

        true_kappa_f.vector().set_local(true_kappa)
        m_f = dl.Function(Vh)
        m_f.vector().set_local(reconstruction_ROM)

        nb.multi1_plot([true_kappa_f, m_f], ["True diffusion", "Solution ROM"], vmax=1.05*np.max(true_kappa[:]), vmin=0.95*np.min(true_kappa[:]))
        plt.show()
    ####################################################################
    # Learnt corrective term 
    create_dataset = False

    if create_dataset:
        prior_mean = prior.mean
        dataset_size = 10000
        parameter_values = np.zeros((dataset_size, Vh.dim()))
        state_values = np.zeros((dataset_size, len(simulation_times), Vh.dim()))
        qoi_values = np.zeros((dataset_size, len(observation_times), targets.shape[0]))
        reduced_state_values = np.zeros((dataset_size, len(simulation_times), basis_size))
        reduced_qoi_values = np.zeros((dataset_size, len(observation_times), targets.shape[0]))
        qoi_bounds = np.zeros((dataset_size, len(observation_times), targets.shape[0]))

        print(f"Size of state samples in mb: {round(state_values.nbytes/(1024 * 1024))}")

        sampling_idx = 0

        while sampling_idx < dataset_size:
            print(f"Sampling index: {sampling_idx} out of {dataset_size}")
            noise = dl.Vector()
            prior.init_vector(noise, "noise")
                
            kappa = dl.Vector()
            prior.init_vector(kappa, 0)
                
            #TODO: Change prior mean
            while True:
                parRandom.normal(1., noise)
                mean_val = np.exp(np.random.uniform(np.log(1e-3), np.log(1e-2)))
                prior.mean = dl.interpolate(dl.Constant(mean_val), Vh).vector()
                prior.sample(noise, kappa)
                if np.min(sampled_values) > 1e-4:
                    break

            true_kappa_f.vector().set_local(kappa)
            #  nb.plot(true_kappa_f); plt.show()

            snapshots = problem.generate_vector(STATE)
            adj_snapshots = problem.generate_vector(ADJOINT)
            x_snapshots = [snapshots, kappa, adj_snapshots]
            problem.solveFwd(x_snapshots[STATE], x_snapshots)

            parameter_values[sampling_idx, :] = kappa[:]

            for time_idx, state in enumerate(x_snapshots[STATE].data):
                state_values[sampling_idx, time_idx, :] = state[:] 

            misfit.observe(x_snapshots, misfit.data)

            out_of_bounds = False
            for time_idx, observation in enumerate(misfit.data.data):
                if np.max(observation[:]) > 1e2:
                    out_of_bounds = True
                qoi_values[sampling_idx, time_idx, :] = observation[:]
            if out_of_bounds:
                print("Large value in QoI encountered. Resampling...")
                continue

            r_snapshots = problem.generate_vector("REDUCED_STATE")
            x_r_snapshots = [r_snapshots, kappa, None]
            problem.solveReducedFwd(x_r_snapshots[STATE], x_r_snapshots)

            for time_idx, state in enumerate(x_r_snapshots[STATE].data):
                reduced_state_values[sampling_idx, time_idx, :] = state[:] 

            x = [problem.u_tildes, kappa, None]
            misfit.observe(x, misfit.data)

            for time_idx, observation in enumerate(misfit.data.data):
                reduced_qoi_values[sampling_idx, time_idx, :] = observation[:]

            is_bound_valid = problem.solveAdj(x_snapshots[ADJOINT], x_snapshots)

            if not is_bound_valid:
                # Erroneous solution, retry
                continue
            #  qoi_bounds[sampling_idx, :, :] = problem.qoi_bounds[:, :]
            qoi_bounds[sampling_idx, :, :] = problem.approx_qoi_bounds[:, :]
            sampling_idx += 1

        np.save('parameter_samples.npy', parameter_values)
        np.save('state_samples.npy', state_values)
        np.save('qoi_samples.npy', qoi_values)
        np.save('reduced_basis.npy', basis)
        np.save('reduced_state_samples.npy', reduced_state_values)
        np.save('reduced_qoi_samples.npy', reduced_qoi_values)
        np.save('qoi_bounds.npy', qoi_bounds)

        del parameter_values
        del state_values
        del qoi_values
        del reduced_state_values
        del reduced_qoi_values
        del qoi_bounds

        prior.mean = prior_mean

    ####################################################################


    if debug:
        if rank == 0:
            print(sep, "Test the gradient and the Hessian of the model", sep)
        m0 = true_kappa.copy()

        # Use hippylib to perform the gradient and Hessian check
        n_eps = 24
        eps_begin_idx = np.ceil(np.log(0.001)/np.log(0.5)) # hippylib finite differencing isn't smart
        eps = np.power(.5, np.arange(eps_begin_idx, n_eps+eps_begin_idx))
        modelVerify(problem, m0, is_quadratic=True,
                misfit_only=False,  verbose=(rank == 0), eps=eps)
        plt.show()

    # Due to stellar naming conventions adopted by Hippylib, below means the true full Hessian
    # not the 'reduced' Hessian. This can be changed to evaluate the Gauss-Newton Hessian by changing
    # the member variable inside `problem`
    H = ReducedHessian(problem, misfit_only=False)

    if rank == 0:
        print(sep, "Solve the optimization using inexact Newton CG", sep)

    # Set the starting parameter to be the prior mean
    [u, m, p] = problem.generate_vector()
    m.set_local(prior.mean)
    problem.solveFwd(u, [u, m, p])
    problem.solveAdj(p, [u, m, p])
    mg = problem.generate_vector(PARAMETER)
    grad_norm = problem.evalGradientParameter([u, m, p], mg)
 
    if rank == 0:
        print("(g,g) = ", grad_norm)

    # Define parameters for the optimization
    tol = 1e-8
    c = 1e-4
    max_iter = 60
    plot_on = False

    # initialize iteration counters
    iter_count = 1
    total_cg_iter = 0
    converged = False

    # initializations
    g, m_delta = dl.Vector(), dl.Vector()
    prior.init_vector(m_delta,0)
    prior.init_vector(g,0)

    m_prev = dl.Function(problem.Vh[PARAMETER])

    [cost_old, misfit_old, reg_old] = problem.cost([u, m, p])
    print( "Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||       alpha  tolcg" )

    while iter_count < max_iter and not converged:
        problem.solveFwd(u, [u, m, p])
        problem.solveAdj(p, [u, m, p])
        grad_norm = problem.evalGradientParameter([u, m, p], mg)

        # set the CG tolerance (use Eisenstat–Walker termination criterion)
        if iter_count == 1:
            grad_norm_ini = grad_norm
        tol_cg = min(0.5, np.sqrt(grad_norm/grad_norm_ini))
    
        # Use Gauss-Newton approximation in the first few iterations. TODO: Currently turned off
        problem.gauss_newton_approx = (iter_count < 1)
        H = ReducedHessian(problem, misfit_only=False)

        solver = CGSolverSteihaug()
        solver.set_operator(H)
        solver.set_preconditioner(prior.Rsolver)
        solver.parameters["rel_tolerance"] = tol_cg
        solver.parameters["zero_initial_guess"] = True
        solver.parameters["print_level"] = -1

        # solve the Newton system H m_delta = - grad(m)
        solver.solve(m_delta, -mg)
        total_cg_iter += H.ncalls

        # Perform Armijo line search
        alpha = 1.0
        descent = False
        no_backtrack = 0
        m_prev.vector().set_local(m)
        while descent == 0 and no_backtrack < 18:
            m.axpy(alpha, m_delta )

            # solve the state/forward problem
            problem.solveFwd(u, [u,m,p])

            # evaluate cost
            [cost_new, misfit_new, reg_new] = problem.cost([u, m, p])

            # check if Armijo conditions are satisfied
            if cost_new < (cost_old + alpha * c * mg.inner(m_delta)):
                cost_old = cost_new
                descent = True
            else:
                no_backtrack += 1
                alpha *= 0.5
                m.set_local(m_prev.vector())  # reset a

        if not descent:
            print("Line search failed. No descent achieved in 18 backtracking steps")
            if debug:
                current_parameter = dl.Function(Vh)
                current_parameter.vector().set_local(m)
                nb.plot(current_parameter)
                plt.show()
            break

        # calculate sqrt(-G * D)
        graddir = np.sqrt(-mg.inner(m_delta))

        # Print the true error for debugging purposes
        print(f"True relative error: {dl.norm(m - true_kappa)/dl.norm(true_kappa)}")

        sp = ""
        print( "%2d %2s %2d %3s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %5.3e %1s %5.3e" % \
            (iter_count, sp, H.ncalls, sp, cost_new, sp, misfit_new, sp, reg_new, sp, \
             graddir, sp, grad_norm, sp, alpha, sp, tol_cg) )
        
        # check for convergence
        if grad_norm < tol and iter_count > 1:
            converged = True
            print(f"Newton's method converged in {iter_count} iterations")
            print(f"Total number of CG iterations: {total_cg_iter}")
            
        iter_count += 1
    
    if not converged:
        print( "Newton's method did not converge in ", max_iter, " iterations" )
    
    true_kappa_f = dl.Function(Vh)
    true_kappa_f.vector().set_local(true_kappa)
    m_f = dl.Function(Vh)
    m_f.vector().set_local(m)
    nb.multi1_plot([true_kappa_f, m_f], ["True diffusion", "Solution"], vmax=1.05*np.max(true_kappa[:]), vmin=0.95*np.min(true_kappa[:]))
    plt.show()
