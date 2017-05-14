workspace()
using Boltzmann
using MNIST
import HTreeRBM


function run_mnist()
  # Set parameters
  Epochs = 80  # Run MNIST dataset over 80 epochs, an epoch is characterized by a number of iterations
  HiddenUnits = 500 # We set the number of hidden units and we have only one hidden layer 

  # Get MNIST training data
  X, labels = traindata() # Access the entire data set at once
  VisUnits = size(X[:,1],1) # Get the training examples (the first column of the matrix)
  HTreeRBM.binarize!(X;level=0.5)  # Create a binary image from image X using the threshold value level.

  # Split sets
  TrainSize = 2000
  ValidSize = 500
  TrainSet = X[:,1:TrainSize]
  ValidSet = X[:,(TrainSize+1):(TrainSize+ValidSize)]

  # Initialize Model
  crbm = HTreeRBM.HTRBM(VisUnits,HiddenUnits; momentum=0.1, sigma=0.1, sigma_hid=0.1)

  #HTreeRBM.init_vbias!(crbm,TrainSet)
  # Training ...
  HTreeRBM.fit(crbm, TrainSet, n_iter=Epochs; lr=0.1)

  return crbm;
end

myrbm = run_mnist()
