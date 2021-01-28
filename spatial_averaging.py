from dolfin import *
from hippylib import *
from matplotlib import pyplot as plt
import numpy as np

def get_measures(mesh, nx):
    '''Get measures of each sub-domain assuming a unit square is divided into
    squares of equal side length nx'''
    domains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, mesh.domains())

    #  plot(domains); plt.show()
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    return (dx, ds)

class Diffusivity(UserExpression):
    def __init__(self, markers, vals, **kwargs):
        super().__init__(**kwargs)
        self.markers = markers
        self.vals = vals
    def eval_cell(self, values, x, cell):
        mark = self.markers[cell.index]
        values[0] = self.vals[mark-1]

class Kappa(UserExpression):
    def __init__(self, avgs, **kwargs):
        super(Kappa, self).__init__(**kwargs)
        self.avgs = avgs
    def eval(self, value, x):
        idx = int((x[0]-DOLFIN_EPS)/0.25)*4 + int((x[1]-DOLFIN_EPS)/0.25)
        value[0] = self.avgs[idx]
    def value_shape(self):
        return ()

def averaged_params_to_func(averaged_params, dx, Vh):
    ''' Convert piece-wise constant function to function in the given FunctionSpace
    Arguments:
        averaged_params - piece-wise constant function values
        dx - FEniCS measure with appropriate subdomains marked
        Vh - FunctionSpace to interpolate to
    '''
    f = interpolate(Kappa(averaged_params), Vh)
    return f

def get_averaging_operator(Vh, dx, nx):
    ''' Obtain linear operator which acts on nodal values of the parameters
    to obtain spatially averaged parameter values '''
    z = TestFunction(Vh)
    operator = np.zeros((nx*nx, Vh.dim()))
    for i in range(nx * nx):
        area = assemble(Constant(1.0) * dx(i+1))
        averaging_op = assemble(z * dx(i+1))/area
        operator[i, :] = averaging_op[:]
    return operator
