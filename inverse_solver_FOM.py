from hippylib import *
from mshr import *
import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, mesh, Vh, prior, misfit, simulation_times, velocity, u_0, gls_stab=False):
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
        '''

        # Set member variables describing the problem
        self.mesh = mesh
        self.Vh = Vh
        self.prior = prior
        self.misfit = misfit
        self.u_0 = u_0

        # Zero boundary conditions TODO: Change these such that they are passed in during initialization
        self.bc_state_dirichlet = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), no_slip_boundary)
        self.bc_adjoint_dirichlet = dl.DirichletBC(Vh[ADJOINT], dl.Constant(0.0), no_slip_boundary)
        self.left_inflow = dl.DirichletBC(Vh[STATE], dl.Constant(0.5), left_slit_boundary)

        # Assume constant timestepping
        self.simulation_times = simulation_times
        dt = simulation_times[1] - simulation_times[0]

        # Trial and Test functions for the weak forms
        u_trial = dl.TrialFunction(Vh[STATE])
        p_trial = dl.TrialFunction(Vh[ADJOINT])
        u_test = dl.TestFunction(Vh[STATE])
        p_test = dl.TestFunction(Vh[ADJOINT])

        # Functions to be populated for time stepping
        self.u_old = dl.Function(Vh[STATE])
        self.p_old = dl.Function(Vh[ADJOINT])
        self.kappa = dl.Function(Vh[PARAMETER])

        dt_expr = dl.Constant(dt)

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
        N_varf = (ufl.inner(ufl.exp(self.kappa) * ufl.grad(u_trial), ufl.grad(u_test)) \
                + ufl.inner(velocity, ufl.grad(u_trial)) * u_test) * ufl.dx
        Nt_varf = (ufl.inner(ufl.exp(self.kappa) * ufl.grad(p_test), ufl.grad(p_trial)) \
                + ufl.inner(velocity, ufl.grad(p_test)) * p_trial) * ufl.dx

        stab_varf = tau*ufl.inner(r_trial, r_test) * ufl.dx

        # LHS variational form to be solved at each time step
        self.L_varf = M_varf + dt_expr * N_varf + stab_varf
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
        self.grad_form = ufl.inner(self.k_hat * ufl.exp(self.kappa) * ufl.grad(self.solved_u), ufl.grad(self.solved_p)) * ufl.dx

        # Setup required storage for Hessian evaluation
        # Refer this guide for a reference: http://g2s3.com/labs/notebooks/Poisson_INCG.html
        self.u_s = TimeDependentVector(self.simulation_times) # Solved solution values
        self.u_s.initialize(self.M, 0)
        self.p_s = TimeDependentVector(self.simulation_times) # Solved adjoint values
        self.p_s.initialize(self.M, 0)
        self.k_first_var = dl.Function(Vh[PARAMETER]) # First variation direction of parameter
        self.C_form = ufl.inner(self.k_first_var * ufl.exp(self.kappa) * ufl.grad(self.solved_u), \
                ufl.grad(u_test)) * ufl.dx
        self.W_um_form = ufl.inner(self.k_first_var * ufl.exp(self.kappa) * ufl.grad(p_test), \
                ufl.grad(self.solved_p)) * ufl.dx
        self.W_mm_form = ufl.inner(\
                self.k_first_var * self.k_hat * ufl.exp(self.kappa) * ufl.grad(self.solved_u),\
                ufl.grad(self.solved_p)) * ufl.dx

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

        u = dl.Function(self.Vh[STATE])

        for t in self.simulation_times[1::]:
            dl.solve(self.L_varf == self.L_rhs_varf, u, bcs=[self.bc_state_dirichlet, self.left_inflow])
            out.store(u.vector(), t)
            self.u_s.store(u.vector(), t) #TODO Fix duplicate storage
            self.u_old.assign(u)

    def solveAdj(self, out, x):
        '''Solve adjoint problem backwards in time and store in out '''
        out.zero()
        self.p_s.zero()
        self.kappa.vector().set_local(x[PARAMETER])

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

        for t in self.simulation_times[::-1]:
            Lt, rhs = dl.assemble_system(self.Lt_varf, self.Lt_rhs_varf, bcs=self.bc_adjoint_dirichlet)
            grad_state.retrieve(grad_state_snap, t)
            rhs.axpy(-0.1, grad_state_snap)

            dl.solve(Lt, p, rhs)
            self.p_old.vector().set_local(p)
            out.store(p, t)
            self.p_s.store(p, t) #TODO Fix duplicate storage

    def evalGradientParameter(self, x, mg, misfit_only=False):
        '''Evaluate gradient with respect to parameters'''
        self.prior.init_vector(mg, 1)
        mg.zero()
        if misfit_only == False:
            dm = x[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)

        for t in simulation_times[1::]:
            x[STATE].retrieve(self.solved_u.vector(), t)
            x[ADJOINT].retrieve(self.solved_p.vector(), t)
            mg.axpy(1., dl.assemble(self.grad_form))

        g = dl.Vector()
        self.M.init_vector(g,1)
        
        self.prior.Msolver.solve(g,mg)
        
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
        self.k_first_var.vector().set_local(dm)
        for t in simulation_times[1::]:
            self.u_s.retrieve(self.solved_u.vector(), t)
            self.p_s.retrieve(self.solved_p.vector(), t)
            out.axpy(1., dl.assemble(self.W_mm_form))

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

def left_boundary(x, on_boundary):
    '''In-flow boundary from the left'''
    return x[0] < dl.DOLFIN_EPS

def no_slip_boundary(x, on_boundary):
    '''Boundary expect the left and right edges where velocity and concentrate is zero'''
    return on_boundary and \
        ((x[0] > dl.DOLFIN_EPS and x[0] < 2.2 - dl.DOLFIN_EPS)\
             or (x[1] < dl.DOLFIN_EPS and x[1] > 0.41 - dl.DOLFIN_EPS))

def left_slit_boundary(x, on_boundary):
    '''In-flow boundary from the left through a slit'''
    return (x[0] < dl.DOLFIN_EPS and (x[1] > 0.1 and x[1] < 0.3))

def computeVelocityField(mesh, plot_velocity=False):
    '''Solves the steady Stokes flow given the geometry and the in-flow conditions.
    This velocity field is computed from the DFG benchmark
    http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
    '''

    Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(2000)

    g = dl.Expression(('4.0 * U * x[1] * (W - x[1])/ (W * W)', '0.0'), W=0.41, U=0.3, degree=1)
    bc_inflow = dl.DirichletBC(XW.sub(0), g, left_boundary)
    
    bc_noslip = dl.DirichletBC(XW.sub(0), dl.Expression(('0.0','0.0'), degree=1), no_slip_boundary)
    
    bcs = [bc_inflow, bc_noslip]

    vq = dl.Function(XW)
    (v,q) = ufl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)

    F = ((2./Re) * ufl.inner(ufl.grad(v), ufl.grad(v_test)) + ufl.dot(ufl.dot(ufl.grad(v), v), v_test) \
           - (q * ufl.div(v_test)) + ( ufl.div(v) * q_test) ) * ufl.dx

    dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                         {"relative_tolerance":1e-4, "maximum_iterations":100}})

    vh = dl.project(v,Xh)
    vh_norm = dl.project(dl.sqrt(dl.inner(v,v)), Wh)
    qh = dl.project(q,Wh)

    if plot_velocity:
        plt.figure(figsize=(30,10))
        nb.plot(vh, subplot_loc=311, mytitle="Velocity")
        nb.plot(vh_norm, subplot_loc=312, mytitle="Velocity magnitude")
        nb.plot(qh, subplot_loc=313,mytitle="Pressure")
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
    N_circle = 16
    N_bulk = 64

    # Define domain
    center = dl.Point(0.2, 0.2)
    radius = 0.05
    L = 2.2
    W = 0.41
    geometry = Rectangle(dl.Point(0.0, 0.0), dl.Point(L, W)) - Circle(center, radius, N_circle)

    # Build mesh
    mesh = generate_mesh(geometry, N_bulk)
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())

    velocity = computeVelocityField(mesh, debug)

    # Function space for state, adjoint, and parameter variables are chosen to be the same
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

    if rank == 0:
        print(sep, "Set up the mesh and finite element spaces.\n",
              "Compute wind velocity", sep)

    ndofs = Vh.dim()
    if rank == 0:
        print("Number of dofs: {0}".format(ndofs))

    if rank == 0:
        print(sep, "Set up Prior Information and model", sep)

    ic_expr = dl.Expression(
        'std::min(0.9, std::exp(-1000*(std::pow(x[0]-0.05,2) +  std::pow((x[1]-0.2)/3,2))) \
                + std::exp(-1000*(std::pow(x[0]-1.0,2) +  std::pow((x[1]-0.2)/3,2))) + \
                std::exp(-1000*(std::pow(x[0]-0.6,2) +  std::pow((x[1]-0.2)/3,2))))',
        element=Vh.ufl_element())
    #  ic_expr = dl.Expression(
        #  'std::min(0.9, std::exp(-1000*(std::pow(x[0]-0.05,2) +  std::pow((x[1]-0.2)/3,2))))',
        #  element=Vh.ufl_element())

    u_0 = dl.interpolate(ic_expr, Vh)
    if debug:
        nb.plot(u_0, mytitle="Initial condition interpolated")
        plt.show()

    # Gaussian priors in infinite dimensions are created following this guide 
    # http://g2s3.com/labs/notebooks/Gaussian_priors.html and using hippylib. 
    # The values below are chosen to have a correlation length ~ 2.0 and pointwise variance ~ 3.0
    gamma = 4.0
    delta = 0.66
    prior = BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)

    prior.mean = dl.interpolate(dl.Constant(np.log(0.005)), Vh).vector()
    true_kappa = dl.interpolate(dl.Constant(np.log(0.002)), Vh).vector()

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
            s_v = sample[:]
            sample.set_local(np.exp(s_v))
            ss.append(vector2Function(sample, Vh))
                
        nb.multi1_plot(ss[0:3], ["sample 1", "sample 2", "sample 3"])
        nb.multi1_plot(ss[3:6], ["sample 4", "sample 5", "sample 6"])
        plt.show()

    if rank == 0:
        print( "Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma, 2))


    # Define simulation and observation times. Observations are made every other time step after t_init
    t_init = 0.
    t_final = 12.
    t_1 = 1.
    dt = .1
    observation_dt = .2

    simulation_times = np.arange(t_init, t_final+.5*dt, dt)
    observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)

    # Define observation targets. Chosen randomly on the domain
    targets = np.random.uniform([0.0, 0.0], [L, W], (200, 2))
    idxs = (np.power((targets[:, 0] - 0.2), 2) + np.power((targets[:,1] - 0.2), 2)) <= (radius * radius)
    targets = np.delete(targets, idxs, 0)
    if rank == 0:
        print("Number of observation points: {0}".format(targets.shape[0]))

    # Observation misfit defined to be pointwise over space and time
    misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)

    problem = TimeDependentAdvectionDiffusion(
        mesh, [Vh, Vh, Vh], prior, misfit, simulation_times, velocity, u_0, False)

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

    if rank == 0:
        print(sep, "Test the gradient and the Hessian of the model", sep)
    m0 = true_kappa.copy()

    # Use hippylib to perform the gradient and Hessian check
    modelVerify(problem, m0, is_quadratic=True,
                misfit_only=False,  verbose=(rank == 0))
    if debug:
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
        print(f"True error: {dl.norm(m - true_kappa)}")

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
    true_kappa_f.vector().set_local(np.exp(true_kappa[:]))
    m_f = dl.Function(Vh)
    m_f.vector().set_local(np.exp(m[:]))
    nb.multi1_plot([true_kappa_f, m_f], ["True diffusion", "Solution"], vmax=1.05*np.exp(np.max(true_kappa[:])), vmin=0.95*np.exp(np.min(true_kappa[:])))
    plt.show()
