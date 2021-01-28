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
from reduced_model import *
from affine_reduced_model import *
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

    #  np.random.seed(123)
    sep = "\n"+"#"*80+"\n"

    # Discretization parameters
    N_size = 40

    # Dimensions of the mesh extremeties
    L = 1.0
    W = 1.0

    # Define domain
    geometry = Rectangle(dl.Point(0.0, 0.0), dl.Point(L, W)) 

    subd_idx = 1
    for ii in range(4):
        for jj in range(4):
            botl = dl.Point(ii * 0.25, jj * 0.25)
            topr = dl.Point((ii+1) * 0.25, (jj+1) * 0.25)
            subd = Rectangle(botl, topr)
            geometry.set_subdomain(subd_idx, subd)
            subd_idx += 1

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

    # Bounds for parameters
    param_lb = 0.001
    param_ub = 0.009

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

    # Prior mean
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
    t_final = 6.0
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

    ####################################################################
    # Synthetic observations

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

    problem_ROM = TimeDependentAdvectionDiffusionReduced(mesh, \
            [Vh, Vh, Vh], prior, misfit, simulation_times, \
            velocity, u_0, observation_times, parameter_basis, False, debug)
    problem_AROM = TimeDependentAdvectionDiffusionAffineReduced(mesh, \
            [Vh, Vh, Vh], prior, misfit, simulation_times, \
            velocity, u_0, observation_times, parameter_basis, False, debug)

    pod_thresh = 1e-13
    n_pod_samples = 10
    param_bounds = np.zeros((problem.n_sq*problem.n_sq, 2))
    param_bounds[:,0] = param_lb
    param_bounds[:,1] = param_ub

    #  basis = PODROM(prior, problem, n_pod_samples, pod_thresh, debug)
    basis = spatially_averaged_PODROM(problem, n_pod_samples, param_bounds, pod_thresh, debug)
    problem.set_reduced_basis(basis)
    problem_ROM.set_reduced_basis(basis)
    problem_AROM.set_reduced_basis(basis)

    if debug:
        if rank == 0:
            print(sep, "Test the gradient of the state reduced model", sep)
        m0 = true_kappa.copy()

        # Use hippylib to perform the gradient and Hessian check
        n_eps = 24
        eps_begin_idx = np.ceil(np.log(0.001)/np.log(0.5)) # hippylib finite differencing isn't smart
        eps = np.power(.5, np.arange(eps_begin_idx, n_eps+eps_begin_idx))
        modelVerify(problem_ROM, m0, is_quadratic=True,
                misfit_only=True,  verbose=(rank == 0), eps=eps, compute_hessian=False)
        plt.show()

        if rank == 0:
            print(sep, "Test the gradient of the parameter and state reduced model", sep)
        m0 = np.dot(problem.averaging_op, true_kappa[:])

        # Use hippylib to perform the gradient and Hessian check
        n_eps = 24
        eps_begin_idx = np.ceil(np.log(0.001)/np.log(0.5)) # hippylib finite differencing isn't smart
        eps = np.power(.5, np.arange(eps_begin_idx, n_eps+eps_begin_idx))
        modelVerify(problem_AROM, m0, is_quadratic=True,
                misfit_only=True,  verbose=(rank == 0), eps=eps, compute_hessian=False)
        plt.show()

        # Check a posteriori error estimate for affine ROM
        u_ROM = problem_AROM.generate_vector()
        u_ROM[PARAMETER][:] = np.dot(problem_AROM.averaging_op, true_kappa[:])
        problem_AROM.solveFwd(u_ROM[STATE], u_ROM)
        problem_AROM.computeErrorEstimate(u_ROM, true_kappa)
        assert np.all(problem_AROM.approx_qoi_bounds > problem.true_qoi_errors)

        # Check a posteriori error estimate for ROM
        u_ROM = problem_ROM.generate_vector()
        u_ROM[PARAMETER].set_local(true_kappa)
        problem_ROM.solveFwd(u_ROM[STATE], u_ROM)
        problem_ROM.computeErrorEstimate(u_ROM)
        assert np.allclose(problem_ROM.true_qoi_errors, problem_ROM.qoi_bounds), \
                "A posteriori error estimate is incorrect."

    def solve_reduced_w_error_estimate(kappa):
        u_s = problem_AROM.generate_vector(STATE)
        p_s = problem_AROM.generate_vector(ADJOINT)
        u_ROM = [u_s, np.dot(problem_AROM.averaging_op, kappa[:]), p_s]
        problem_AROM.solveFwd(u_ROM[STATE], u_ROM)
        problem_AROM.computeErrorEstimate(u_ROM)

    if debug:
        avg_true_kappas = np.dot(problem_AROM.averaging_op, true_kappa[:]) 
        x_r = problem_AROM.generate_vector()
        x_r[PARAMETER] = avg_true_kappas
        problem_AROM.solveFwd(x_r[STATE], x_r)
        nb.show_solution(Vh, u_0.vector(), problem_AROM.u_tildes, mytitle="Affine ROM Sol", \
                times=np.linspace(t_init, t_final, 6))
        plt.show()

    if "reduced_inverse" in sys.argv:
        starting_parameter_values = prior.mean[:]
        #  starting_parameter_values = dl.interpolate(dl.Constant(0.003), Vh).vector()[:]
        print(f'Starting cost: {reduced_cost_function(starting_parameter_values)}')
        bounds = Bounds(0.001, 0.006)
        res = minimize(problem_ROM.cost_function, starting_parameter_values, 
                method='L-BFGS-B', 
                jac=problem_ROM.gradient,
                bounds=bounds,
                options={'ftol':1e-8, 'gtol':1e-8, 'maxls':20, 'iprint':11})
        print(f'\nstatus: {res.success}, message: {res.message}, n_it: {res.nit}')
        print(f'Minimum cost: {res.fun:.3F}')

        reconstruction_ROM = problem_ROM.generate_vector(PARAMETER)
        reconstruction_ROM.set_local(res.x)
        print(f"Relative reconstruction error with ROM forward: {dl.norm(reconstruction_ROM - true_kappa)/dl.norm(true_kappa)}")

        true_kappa_f.vector().set_local(true_kappa)
        m_f = dl.Function(Vh)
        m_f.vector().set_local(reconstruction_ROM)

        nb.multi1_plot([true_kappa_f, m_f], ["True diffusion", "Solution ROM"], vmax=1.05*np.max(true_kappa[:]), vmin=0.95*np.min(true_kappa[:]))
        plt.show()


    if "affine_reduced_inverse" in sys.argv:
        starting_parameter_values = 0.002 * np.ones((problem_AROM.n_sq**2,))
        #  starting_parameter_values = dl.interpolate(dl.Constant(0.003), Vh).vector()[:]
        print(f'Starting cost: {problem_AROM.cost_function(starting_parameter_values)}')
        bounds = Bounds(0.001, 0.006)
        res = minimize(problem_AROM.cost_function, starting_parameter_values, 
                method='L-BFGS-B', 
                jac=problem_AROM.gradient,
                bounds=bounds,
                options={'ftol':1e-8, 'gtol':1e-8, 'maxls':20, 'iprint':11})
        print(f'\nstatus: {res.success}, message: {res.message}, n_it: {res.nit}')
        print(f'Minimum cost: {res.fun:.3F}')

        true_kappa_f.vector().set_local(true_kappa)
        res_f = averaged_params_to_func(res.x, problem_AROM.a_dx, problem_AROM.Vh[PARAMETER])
        print(f"Relative reconstruction error with Affine ROM forward: {dl.norm(res_f.vector() - true_kappa)/dl.norm(true_kappa)}")

        nb.multi1_plot([true_kappa_f, res_f], ["True diffusion", "Solution ROM"], vmax=1.05*np.max(true_kappa[:]), vmin=0.95*np.min(true_kappa[:]))
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
