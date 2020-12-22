from dolfin import *
import numpy as np

class SubBlock(SubDomain):
    '''Defines a square subblock of the domain given bottom left anchor point and side length'''
    def __init__(self, bot_left_x, bot_left_y, side_length, **kwargs):
        self.bot_left_x = bot_left_x
        self.bot_left_y = bot_left_y
        self.side_length = side_length
        super(SubBlock, self).__init__(**kwargs)

    def inside(self, x, on_boundary):
        return between(x[0], (self.bot_left_x, self.bot_left_x + self.side_length)) and \
                between(x[1], (self.bot_left_y, self.bot_left_y + self.side_length))

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
            block = SubBlock(xx, yy, (1.0/nx))
            block.mark(domains, idx)
            boundary = SubBlock(xx, yy, (1.0/nx))
            boundary.mark(boundaries, idx)
            idx += 1


    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    return (dx, ds)

def get_averaging_operator(Vh, dx, nx):
    z = TestFunction(Vh)
    operator = np.zeros((nx*nx, Vh.dim()))
    for i in range(nx * nx):
        area = assemble(Constant(1.0) * dx(i+1))
        averaging_op = assemble(z * dx(i+1))/area
        operator[i, :] = averaging_op[:]
    return operator
