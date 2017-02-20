# caffe-manual-sgd
implement the SGD functionality to update weights in python manually in caffe python instead of using solver.step() function. 
There is a question in stackoverflow discussed about it[1].

to compare to the weight updates after doing solver.step() and that by manually updating the weights, we need to train with gpu as cudnn
may cause slightly different results.

Steps:

1. git clone caffe, and compile it

2. download mnist dataset

  `./data/mnist/get_mnist.sh`
  
3. create lmdb

  `examples/mnist/create_mnist.sh`
  
4. modify "examples/mnist/lenet_solver.prototxt"
  ```
  +random_seed: 52
   # The learning rate policy
  -lr_policy: "inv"
  -gamma: 0.0001
  -power: 0.75
  +lr_policy: "step"
  +gamma: 0.1
  +stepsize: 5000

  -solver_mode: GPU
  +solver_mode: CPU
  ```
  
5. run the test

  `python examples/mnist/train.py`




[1] http://stackoverflow.com/questions/36459266/caffe-python-manual-sgd
