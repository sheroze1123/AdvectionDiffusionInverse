from hippylib import *
import dolfin as dl; dl.set_log_level(40)
import matplotlib.pyplot as plt
import numpy as np
from smt.sampling_methods import LHS
from spatial_averaging import *

def PODROM(prior, problem, samples, pod_thresh, debug=False):
    # Replace with model-constrained adaptive sampling
    # Create reduced-space basis sampling from the prior 
    S_full = None
    n_sim_times = len(problem.simulation_times)
    Vh = problem.Vh[STATE]
    dt = problem.simulation_times[1] - problem.simulation_times[0]

    for prior_sample in range(samples):
        noise = dl.Vector()
        prior.init_vector(noise, "noise")
            
        kappa = dl.Vector()
        prior.init_vector(kappa, 0)
            
        parRandom.normal(1., noise)
        prior.sample(noise, kappa)

        snapshots = problem.generate_vector(STATE)
        x_snapshots = [snapshots, kappa, None]
        problem.solveFwd(x_snapshots[STATE], x_snapshots)

        Y = np.zeros((n_sim_times, Vh.dim()))
        S = np.zeros((Vh.dim(), n_sim_times - 1))
        ii = 0
        for u in snapshots.data:
            weight = dt
            if ii==0 or ii==(n_sim_times-1):
                weight /= 2
            Y[ii, :] = np.sqrt(weight) * u[:]
            if ii > 0:
                S[:, ii-1] = u[:] - problem.u_0.vector()[:]
            ii += 1
        if S_full is None:
            S_full = S
        else:
            S_full = np.column_stack((S_full, S))
            
    UU, SS, VV = np.linalg.svd(S_full)

    if debug:
        plt.semilogy(SS); plt.title("Eigenvalue decay of prior samples")
        plt.show()
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
    return basis

def spatially_averaged_PODROM(problem, n_samples, bounds, pod_thresh=1e-13, debug=False):
    ''' Generate ROM basis assuming parameters are coefficients of a given basis '''
    S_full = None
    n_sim_times = len(problem.simulation_times)
    Vh = problem.Vh[STATE]
    dt = problem.simulation_times[1] - problem.simulation_times[0]
    sampling = LHS(xlimits=bounds)
    parameter_samples = sampling(n_samples)
    kappa = dl.Vector()

    parameter_samples[0,:] = 0.002

    for idx in range(n_samples):
        param = parameter_samples[idx, :]
        kappa = averaged_params_to_func(param, problem.a_dx, Vh).vector()
        snapshots = problem.generate_vector(STATE)
        x_snapshots = [snapshots, kappa, None]
        problem.solveFwd(x_snapshots[STATE], x_snapshots)

        Y = np.zeros((n_sim_times, Vh.dim()))
        S = np.zeros((Vh.dim(), n_sim_times - 1))
        ii = 0
        for u in snapshots.data:
            weight = dt
            if ii==0 or ii==(n_sim_times-1):
                weight /= 2
            Y[ii, :] = np.sqrt(weight) * u[:]
            if ii > 0:
                S[:, ii-1] = u[:] - problem.u_0.vector()[:]
            ii += 1
        if S_full is None:
            S_full = S
        else:
            S_full = np.column_stack((S_full, S))
            
    UU, SS, VV = np.linalg.svd(S_full)

    if debug:
        plt.semilogy(SS); plt.title("Eigenvalue decay of prior samples")
        plt.show()
    basis_size = np.sum(SS > pod_thresh)
    print(f"Number of singular values larger than {pod_thresh}: {basis_size}")
    basis = UU[:, :basis_size]
    print(f"Dimension of reduced basis: {basis.shape[1]}")
    return basis
