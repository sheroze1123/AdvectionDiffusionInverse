import tensorflow
import sys; sys.path.append('../')
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from dolfin import set_log_level; set_log_level(40)

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l1_l2, l2, l1
from tensorflow.keras.initializers import *
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import LearningRateScheduler

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta


def residual_unit(x, activation, n_weights, l1_reg=1e-8, l2_reg=1e-4):
    '''A single residual unit with a skip connection and batch normalization'''
    res = x

    out = BatchNormalization()(x)
    out = activation(out)
    out = Dense(n_weights, activation=None, kernel_regularizer=l1_l2(l1_reg, l2_reg))(out)

    out = BatchNormalization()(x)
    out = activation(out)
    out = Dense(n_weights, activation=None, kernel_regularizer=l1_l2(l1_reg, l2_reg))(out)

    out = add([res, out])
    return out

def parametric_model(
        activation, 
        optimizer, 
        lr, 
        lr_decay, 
        n_hidden_layers, 
        n_weights, 
        batch_size,
        x_train,
        y_train, 
        x_val, 
        y_val,
        n_epochs=100,
        model_return=False):

    input_shape = x_train[0].shape[1]
    output_shape = y_train.shape[1]

    inputs = Input(shape=(input_shape,))
    bound_input = Input(shape=(output_shape,))
    y = Dense(n_weights, input_shape=(input_shape,), activation=None, 
            kernel_regularizer=l1_l2(1e-3, 1e-3))(inputs)
    out = residual_unit(y, activation, n_weights)
    for i in range(1, n_hidden_layers):
        out = residual_unit(out, activation, n_weights)
    out = BatchNormalization()(out)
    out = activation(out)
    #  out = Dense(output_shape, activation='tanh', kernel_initializer=Zeros())(out)
    out = Dense(output_shape, activation='tanh')(out)
    multiplied = Multiply()([out, bound_input])
    model = Model(inputs=[inputs, bound_input], outputs=multiplied)
    #  model = Model(inputs=inputs, outputs=out)
    model.compile(
            loss='mse', 
            optimizer=optimizer(lr=lr, decay=lr_decay), 
            metrics=['mape'])
    history = model.fit(
            x_train, 
            y_train, 
            epochs=n_epochs, 
            batch_size=batch_size,
            validation_data=(x_val, y_val))

    tr_losses = history.history['mape']
    vmapes = history.history['val_mape']

    # Mean Absolute Relative Error is the validation metric
    vmape = vmapes[-1]

    if model_return:
        return vmape, model

    return vmape

initial_learning_rate = 1e-2
def lr_schedule(epoch):
    '''Callback function to schedule learning rate decay'''
    if epoch<=1000:
        return initial_learning_rate
    elif epoch<=2000:
        return initial_learning_rate/10
    elif epoch<=3000:
        return initial_learning_rate/100
    elif epoch<=9000:
        return 1e-7
    else:
        return 1e-7

parameter_values     = np.load('parameter_samples.npy')
state_values         = np.load('state_samples.npy')     
qoi_values           = np.load('qoi_samples.npy')           
reduced_basis        = np.load('reduced_basis.npy')        
reduced_state_values = np.load('reduced_state_samples.npy')
reduced_qoi_values   = np.load('reduced_qoi_samples.npy')
qoi_bounds           = np.load('qoi_bounds.npy')

#  a_idx = (np.max(qoi_values.reshape(10000,qoi_values.shape[1] * qoi_values.shape[2]), axis=1) < 1)
#  parameter_values = parameter_values[a_idx, :]
#  state_values = state_values[a_idx, :, :]
#  qoi_values = qoi_values[a_idx, :, :]
#  reduced_state_values = reduced_state_values[a_idx, :, :]
#  reduced_qoi_values = reduced_qoi_values[a_idx, :, :]
#  qoi_bounds = qoi_bounds[a_idx, :, :]
qoi_errors = qoi_values - reduced_qoi_values

#  mean_parameter_value = np.mean(parameter_values)
#  stdev_parameter_value = np.std(parameter_values)
#  parameter_values = (parameter_values - mean_parameter_value)/stdev_parameter_value

#  mean_qoi_errors = np.mean(qoi_errors)
#  std_qoi_errors = np.std(qoi_errors)
#  qoi_errors = (qoi_errors - mean_qoi_errors)/std_qoi_errors

parameter_dim = parameter_values.shape[1]
qoi_dim = qoi_values.shape[1] * qoi_values.shape[2]
dataset_size = parameter_values.shape[0]
print(f"Dataset size: {dataset_size}")
tr_split = int(0.80 * dataset_size)

parameters_train = parameter_values[:tr_split, :]
parameters_validation = parameter_values[tr_split:, :]
bounds_train = np.zeros((tr_split, qoi_dim))
bounds_validation = np.zeros((dataset_size - tr_split, qoi_dim))
errors_train = np.zeros((tr_split, qoi_dim))
errors_validation = np.zeros((dataset_size - tr_split, qoi_dim))

bounds_slack = 1.0
for idx in range(dataset_size):
    if idx < tr_split:
        errors_train[idx, :] = qoi_errors[idx, :, :].reshape((qoi_dim,))
        bounds_train[idx, :] = bounds_slack * qoi_bounds[idx, :, :].reshape((qoi_dim,))
    else:
        errors_validation[idx-tr_split, :] = qoi_errors[idx, :, :].reshape((qoi_dim,))
        bounds_validation[idx-tr_split, :] = bounds_slack * qoi_bounds[idx, :, :].reshape((qoi_dim,))

space = [Categorical([elu, tanh, relu], name='activation'),
         Categorical([Adam, Adadelta], name='optimizer'),
         Real(1e-7, 1, prior="log-uniform", name='lr'),
         Real(1e-9, 1e-2, prior='log-uniform', name='lr_decay'),
         Integer(1, 6, name='n_hidden_layers'),
         Integer(10, 100, name='n_weights'),
         Integer(4, 500, name='batch_size')]

@use_named_args(space)
def objective(**params):
    return parametric_model(**params, 
            x_train = [parameters_train, bounds_train],
            y_train = errors_train, 
            x_val = [parameters_validation, bounds_validation],
            y_val = errors_validation, 
            n_epochs = 100)

res_gp = gp_minimize(objective, space, n_calls=100, random_state=None)

print("Best score: {}".format(res_gp.fun))
print('''Best parameters:\n
 Activation function: {}
 Optimizer: {}
 Learning rate: {}
 Learning rate decay: {}
 Num. hidden Layers: {}
 Num. weights: {}
 Batch size: {}'''.format(*res_gp.x))

plot_convergence(res_gp, yscale="log")
plt.show()

activation = res_gp.x[0]
optimizer = res_gp.x[1]
opt_lr = res_gp.x[2]
lr_decay = res_gp.x[3]
n_hidden_layers = res_gp.x[4]
n_weights = res_gp.x[5]
batch_size = res_gp.x[6]
n_epochs = 10000

vmape, model = parametric_model(activation, 
        optimizer, 
        opt_lr, 
        lr_decay, 
        n_hidden_layers, 
        n_weights, 
        batch_size,
        [parameters_train, bounds_train],
        errors_train, 
        [parameters_validation, bounds_validation], 
        errors_validation, 
        n_epochs, 
        model_return=True)
