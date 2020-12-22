from hippylib import *
from mshr import *
import dolfin as dl; dl.set_log_level(40)
import ufl

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

    def grad_reduced(self, i, x, out, phi):
        ''' Compute the gradient of the ROM cost function with respecto to i = {STATE, PARAMETER} '''
        out.zero()

        u_r_snapshot = dl.Vector()
        phi.init_vector(u_r_snapshot, 1)
        B_phi = dl.PETScMatrix(dl.as_backend_type(self.B).mat().matMult(phi.mat()))
        
        if i == STATE:
            # Gradient w.r.t state is simply B^T(Bu - d)
            for t in self.observation_times:
                x.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.data.retrieve(self.data_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.data_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                B_phi.transpmult(self.Bu_snapshot, u_r_snapshot)
                out.store(u_r_snapshot, t)
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

