from hippylib import *
from mshr import *
import dolfin as dl; dl.set_log_level(40)
import ufl
import numpy as np
import matplotlib.pyplot as plt
import sys
from petsc4py import PETSc

# Compute MAP estimate using gradients
from scipy.optimize import minimize, Bounds

from observations import *
from model import *
from velocity import *
from prior import *
from rom import *


if __name__ == "__main__":
    try:
        dl.set_log_active(False)
    except:
        pass

    # Possible command line arguments are 
    #   "debug"
    #   "reduced_inverse"
    #   "dataset"

    # Turn on debug mode to plot all quantities and print debug values
    debug = "debug" in sys.argv

    #np.random.seed(123)
    sep = "\n"+"#"*80+"\n"

    # Discretization parameters
    N_size = 40

    # Dimensions of the mesh extremeties
    L = 1.0
    W = 1.0

    # Define domain
    geometry = Rectangle(dl.Point(0.0, 0.0), dl.Point(L, W)) 

    # Build mesh
    mesh = generate_mesh(geometry, N_size)
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())

    # Function space for state, adjoint, and parameter variables are chosen to be the same
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

    velocity = computeVelocityField(mesh, L, W, debug)

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
        visualize_prior(prior, Vh)

    if rank == 0:
        print( "Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma, 2))

    # Define simulation and observation times. Observations are made every other time step after t_init
    t_init = 0.
    t_final = 6.
    t_1 = 0.2
    dt = .1
    observation_dt = .2

    simulation_times = np.arange(t_init, t_final+.5*dt, dt)
    observation_times = simulation_times[round(t_1/dt)::round(observation_dt/dt)]

    # Define observation targets. Chosen randomly on the domain
    ntargets = 20
    targets = np.random.uniform([0.0, 0.0], [L, W], (ntargets, 2))
    if rank == 0:
        print("Number of observation points: {0}".format(targets.shape[0]))

    # Observation misfit defined to be pointwise over space and time
    misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)

    problem = TimeDependentAdvectionDiffusion(mesh, \
            [Vh, Vh, Vh], prior, misfit, simulation_times, \
            velocity, u_0, observation_times, parameter_basis, False, debug)

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
    pod_thresh = 1e-13
    n_pod_samples = 10
    param_bounds = np.zeros((problem.n_sq*problem.n_sq, 2))
    param_bounds[:,0] = 0.001
    param_bounds[:,1] = 0.009

    #  basis = PODROM(prior, problem, n_pod_samples, pod_thresh, debug)
    basis = spatially_averaged_PODROM(problem, n_pod_samples, param_bounds, pod_thresh, debug)
    problem.set_reduced_basis(basis)

    def solve_reduced_w_error_estimate(kappa):
        u_s = problem.generate_vector(STATE)
        p_s = problem.generate_vector(ADJOINT)
        u_FOM = [u_s, kappa, p_s]
        problem.solveFwd(u_FOM[STATE], u_FOM)

        utrue_r = problem.generate_vector("REDUCED_STATE")
        avg_kappas = np.dot(problem.averaging_op, kappa[:])
        x = [utrue_r, avg_kappas, None]
        problem.solveAffineROM(x[STATE], x)
        problem.solveAdj(u_FOM[ADJOINT], u_FOM) # Computes full adjoint and estimates bound on error

    solve_reduced_w_error_estimate(true_kappa)

    utrue_r = problem.generate_vector("REDUCED_STATE")
    ptrue_r = problem.generate_vector("REDUCED_STATE")
    avg_true_kappas = np.dot(problem.averaging_op, true_kappa[:]) 
    x = [utrue_r, avg_true_kappas, ptrue_r]
    problem.solveAffineROM(x[STATE], x)
    if debug:
        nb.show_solution(Vh, u_0.vector(), problem.u_tildes, mytitle="Solution RB", \
                times=np.linspace(t_init, t_final, 6))
        plt.show()
    problem.solveAffineROMAdj(x[ADJOINT], x)
    mg = problem.generate_vector(PARAMETER)
    grad_norm_r = problem.evalGradientParameter(x, mg, use_ROM=True)
    print(f"Norm of the reduced gradient {grad_norm_r}")

    if debug:
        h = problem.generate_vector(PARAMETER)
        parRandom.normal(1., h)
        
        u_r = problem.generate_vector("REDUCED_STATE")
        p_r = problem.generate_vector("REDUCED_STATE")
        avg_true_kappas = np.dot(problem.averaging_op, true_kappa[:]) 
        x_r = [u_r, avg_true_kappas, p_r]
        problem.solveAffineROM(x[0], x)
        problem.solveAffineROMAdj(x[2], x)
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
            problem.solveAffineROM(x_r_plus[0], x_r_plus)
            problem.solveAffineROMAdj(x_r_plus[2], x_r_plus)
            x_plus = [problem.u_tildes, k, problem.p_tildes]
            
            dc = problem.cost(x_plus)[0] - cx[0]
            err_grad[i] = abs(dc/my_eps - grad_xh)

        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check for Reduced Model")
        plt.show()

    def reduced_cost_function(param):
        '''Cost function for reduced problem in numpy'''
        u_r = problem.generate_vector("REDUCED_STATE")
        p_r = problem.generate_vector("REDUCED_STATE")
        k = problem.generate_vector(PARAMETER)
        k.set_local(param)
        x_r = [u_r, k, p_r]
        problem.solveReducedFwd(x_r[STATE], x_r)
        problem.solveReducedAdj(x_r[ADJOINT], x_r)
        x = [problem.u_tildes, k, problem.p_tildes]
        return problem.cost(x)[0]

    def reduced_gradient(param):
        '''Gradient of the cost function for the reduced problem in numpy'''
        u_r = problem.generate_vector("REDUCED_STATE")
        p_r = problem.generate_vector("REDUCED_STATE")
        k = problem.generate_vector(PARAMETER)
        k.set_local(param)
        x_r = [u_r, k, p_r]
        problem.solveReducedFwd(x_r[STATE], x_r)
        problem.solveReducedAdj(x_r[ADJOINT], x_r)
        grad_x = problem.generate_vector(PARAMETER)
        problem.evalGradientParameter(x_r, grad_x, misfit_only=False, use_ROM=True)
        return grad_x[:]

    def affine_ROM_cost_function(param):
        '''Cost function for double reduced problem where parameters are spatial averages'''
        u_r = problem.generate_vector("REDUCED_STATE")
        x_r = [u_r, param, None]
        problem.solveAffineROM(x_r[STATE], x_r)
        x = [problem.u_tildes, \
             averaged_params_to_func(param, problem.a_dx, problem.Vh[PARAMETER]), \
             None]
        return problem.cost(x)[0]

    def affine_ROM_gradient(param):
        '''Gradient of the cost function for the affinely decomposed reduced problem in numpy'''
        u_r = problem.generate_vector("REDUCED_STATE")
        p_r = problem.generate_vector("REDUCED_STATE")
        x_r = [u_r, param, p_r]
        problem.solveAffineROM(x_r[STATE], x_r)
        problem.solveAffineROMAdj(x_r[ADJOINT], x_r)
        return problem.evalGradientAveragedParameter(x_r, grad_x, misfit_only=False)

    solve_reduced_inverse = "reduced_inverse" in sys.argv

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
    create_dataset = "dataset" in sys.argv

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
            x_r_snapshots = [r_snapshots, np.dot(problem.averaging_op, kappa[:]), None]
            problem.solveAffineROM(x_r_snapshots[STATE], x_r_snapshots)

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
