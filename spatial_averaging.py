from dolfin import *
from hippylib import *
from matplotlib import pyplot as plt
import numpy as np

class SubBlock(SubDomain):
    '''Defines a square subblock of the domain given bottom left anchor point and side length'''
    def __init__(self, bot_left_x, bot_left_y, side_length, hmin, **kwargs):
        self.bot_left_x = bot_left_x
        self.bot_left_y = bot_left_y
        self.side_length = side_length
        self.hmin = hmin
        super(SubBlock, self).__init__(**kwargs)

    def inside(self, x, on_boundary):
        e = self.hmin 
        bw = between(x[0], (self.bot_left_x - e, self.bot_left_x + self.side_length + e)) and \
                between(x[1], (self.bot_left_y - e, self.bot_left_y + self.side_length + e))
        return bw

def get_measures(mesh, nx):
    '''Get measures of each sub-domain assuming a unit square is divided into
    squares of equal side length nx'''
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)

    idx = 1
    for xx in np.linspace(0.0, 1.0 - 1.0/nx, nx):
        for yy in np.linspace(0.0, 1.0 - 1.0/nx, nx):
            block = SubBlock(xx, yy, (1.0/nx), mesh.hmin())
            block.mark(domains, idx)
            boundary = SubBlock(xx, yy, (1.0/nx), mesh.hmin())
            boundary.mark(boundaries, idx)
            idx += 1

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    return (dx, ds)

def averaged_params_to_func(averaged_params, dx, Vh):
    ''' Convert piece-wise constant function to function in the given FunctionSpace
    Arguments:
        averaged_params - piece-wise constant function values
        dx - FEniCS measure with appropriate subdomains marked
        Vh - FunctionSpace to interpolate to
    '''
    V0 = FunctionSpace(Vh.mesh(), 'DG', 0)
    pw_f  = Function(V0)
    subdomains = dx.subdomain_data()
    for cell_no in range(len(subdomains.array())):
        subdomain_no = int(subdomains.array()[cell_no])
        pw_f.vector()[cell_no] = averaged_params[subdomain_no-1]

    f = interpolate(pw_f, Vh)
    return interpolate(pw_f, Vh)

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
