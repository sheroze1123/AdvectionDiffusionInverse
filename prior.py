from hippylib import *
import dolfin as dl; dl.set_log_level(40)
import matplotlib.pyplot as plt

def visualize_prior(prior, Vh):
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
        
    sample = dl.Vector()
    prior.init_vector(sample, 0)
        
    ss = []
        
    for i in range(6):
        parRandom.normal(1., noise)
        prior.sample(noise, sample)
        sampled_values = sample[:]; sample.set_local(sampled_values)
        ss.append(vector2Function(sample, Vh))
            
    nb.multi1_plot(ss[0:3], ["sample 1", "sample 2", "sample 3"])
    nb.multi1_plot(ss[3:6], ["sample 4", "sample 5", "sample 6"])
    plt.show()
