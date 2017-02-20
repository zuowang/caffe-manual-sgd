import caffe
caffe.set_device(0)
caffe.set_mode_cpu()
import numpy as np

niter = 10000
solver = None
solver = caffe.SGDSolver('examples/mnist/lenet_solver.prototxt')

# Automatic SGD: TEST2
solver.step(niter)
# save the weights to compare later
w_solver_step = solver.net.layers[1].blobs[0].data.copy()
b_solver_step = solver.net.layers[1].blobs[1].data.copy()

# Manual SGD: TEST1
solver = None
solver = caffe.SGDSolver('examples/mnist/lenet_solver.prototxt')
base_lr = 0.01
momentum = 0.9
weight_decay = 0.0005
lr_w_mult = 1
lr_b_mult = 2
gamma = 0.1
stepsize = 5000

momentum_hist = {}
for layer in solver.net.params:
    m_w = np.zeros_like(solver.net.params[layer][0].data)
    m_b = np.zeros_like(solver.net.params[layer][1].data)
    momentum_hist[layer] = [m_w, m_b]

for it in range(1, niter+1):
    solver.net.forward()  # fprop
    solver.net.backward()  # bprop
    for layer in solver.net.params:
        momentum_hist[layer][0] = momentum_hist[layer][0] * momentum + (solver.net.params[layer][0].diff + weight_decay *
                                                       solver.net.params[layer][0].data) * base_lr * lr_w_mult
        momentum_hist[layer][1] = momentum_hist[layer][1] * momentum + (solver.net.params[layer][1].diff + weight_decay *
                                                       solver.net.params[layer][1].data) * base_lr * lr_b_mult
        solver.net.params[layer][0].data[...] -= momentum_hist[layer][0]
        solver.net.params[layer][1].data[...] -= momentum_hist[layer][1]
        solver.net.params[layer][0].diff[...] *= 0
        solver.net.params[layer][1].diff[...] *= 0
    base_lr = base_lr * np.power(gamma, (np.floor(it / stepsize)))
    
# save the weights to compare later
w_fwdbwd_update = solver.net.layers[1].blobs[0].data.copy()
b_fwdbwd_update = solver.net.layers[1].blobs[1].data.copy()

# Compare
print "after iter", niter, ": weight diff: ", np.linalg.norm(w_solver_step - w_fwdbwd_update), "and bias diff:", np.linalg.norm(b_solver_step - b_fwdbwd_update)
