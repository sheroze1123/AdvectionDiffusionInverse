from hippylib import *
from mshr import *
import dolfin as dl; dl.set_log_level(40)
import ufl
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

from spatial_averaging import *

# Dimensions of the mesh extremeties
L = 1.0
W = 1.0

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

        # Spatial averaging affine decomposition
        self.n_sq = 4
        self.a_dx, self.a_ds = get_measures(mesh, self.n_sq)
        self.averaged_params = [Constant(1.0) for i in range(self.n_sq * self.n_sq)]
        self.averaged_L_varf = None
        self.averaged_S_varf = None

        for i in range(self.n_sq * self.n_sq):
            M_varf = ufl.inner(u_trial, u_test) * self.a_dx(i)
            N_varf = (ufl.inner(self.averaged_params[i] * ufl.grad(u_trial), ufl.grad(u_test))\
                    + ufl.inner(velocity, ufl.grad(u_trial)) * u_test) * self.a_dx(i)
            
            if i==0:
                self.averaged_L_varf = M_varf + dt_expr * N_varf
                self.averaged_S_varf = dt_expr * ufl.inner(source, u_test) * self.a_dx(i)
            else:
                self.averaged_L_varf += M_varf + dt_expr * N_varf
                self.averaged_S_varf += dt_expr * ufl.inner(source, u_test) * self.a_dx(i)

        self.averaging_op = get_averaging_operator(self.Vh[PARAMETER], self.a_dx, self.n_sq)

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
        averaged_params = np.dot(self.averaging_op, self.kappa.vector()[:])
        for p in range(len(self.averaged_params)):
            self.averaged_params[p].assign(averaged_params[p])
        affine_L = dl.as_backend_type(dl.assemble(self.averaged_L_varf)).mat()
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
            for t in self.simulation_times[1::]:
                self.u_tildes.retrieve(self.solved_u.vector(), t)
                self.L_p_tildes.retrieve(self.solved_p.vector(), t)
                mg.axpy(1., dl.assemble(self.grad_form))
                self.L_u_tildes.retrieve(self.solved_u.vector(), t)
                self.p_tildes.retrieve(self.solved_p.vector(), t)
                mg.axpy(1., dl.assemble(self.grad_form))
        else:
            for t in self.simulation_times[1::]:
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
        for t in self.simulation_times[1::]:
            self.u_s.retrieve(self.solved_u.vector(), t)
            dp.retrieve(self.solved_p.vector(), t)
            out.axpy(1., dl.assemble(self.grad_form))

    def applyWmu(self, du, out):
        out.zero()
        for t in self.simulation_times[1::]:
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
