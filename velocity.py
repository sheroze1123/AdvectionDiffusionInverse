from hippylib import *
import dolfin as dl; dl.set_log_level(40)
import numpy as np
import ufl
import matplotlib.pyplot as plt

def v_boundary(x, on_boundary):
    '''Velocity boundary prescribed except on right-corner'''
    return on_boundary and x[0] < 1.0 - dl.DOLFIN_EPS


def computeVelocityField(mesh, L, W, plot_velocity=False):
    '''Solves the steady Stokes flow given the geometry and the in-flow conditions.
    Arguments:
        L - domain length
        W - domain width
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
