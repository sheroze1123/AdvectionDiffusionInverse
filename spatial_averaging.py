from dolfin import *
from hippylib import *
from matplotlib import pyplot as plt
import numpy as np

#  class SubBlock(SubDomain):
    #  '''Defines a square subblock of the domain given bottom left anchor point and side length'''
    #  def __init__(self, bot_left_x, bot_left_y, side_length, hmin, **kwargs):
        #  self.bot_left_x = bot_left_x
        #  self.bot_left_y = bot_left_y
        #  self.side_length = side_length
        #  self.hmin = hmin
        #  super(SubBlock, self).__init__(**kwargs)

    #  def inside(self, x, on_boundary):
        #  e = self.hmin 
        #  bw = between(x[0], (self.bot_left_x - e, self.bot_left_x + self.side_length + e)) and \
                #  between(x[1], (self.bot_left_y - e, self.bot_left_y + self.side_length + e))
        #  return bw

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
    #  pw_f  = Function(Vh)
    #  subdomains = dx.subdomain_data()
    #  for cell_no in range(len(subdomains.array())):
        #  subdomain_no = int(subdomains.array()[cell_no])
        #  print(f"subd no: {subdomain_no}")
        #  pw_f.vector()[cell_no] = averaged_params[subdomain_no-1]

    #  return pw_f

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

def avgf(avgs, Vh, dx, nx):
    z = TestFunction(Vh)
    operator = np.zeros((nx*nx, Vh.dim()))
    varf = None
    for i in range(nx * nx):
        if varf is None:
            varf = z * avgs[i] * dx(i+1) 
        else:
            varf += z * avgs[i] * dx(i+1)
        #  area = assemble(Constant(1.0) * dx(i+1))
        #  averaging_op = assemble(z * dx(i+1))/area
        #  operator[i, :] = averaging_op[:]
    return assemble(varf)
    #  return operator
