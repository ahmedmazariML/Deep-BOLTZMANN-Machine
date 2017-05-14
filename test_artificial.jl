workspace()

using HDF5, JLD
import HTreeRBM
import HTreeRBM.gibbs
import HTreeRBM.BPTree

function run_artificial_data()
  # Set parameters

  Epochs = 3#150 gives a better performance but very slow to train HRBM with my machine 
  HiddenUnits = 500 # We set the number of hidden units and we have only one hidden layer 



 

	X = h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/rbm_input_new_new.h5", "r") do file
	    read(file, "a")
	end


  VisUnits = size(X[:,1],1) # Get the training examples (the first column of the matrix)
  

  # Split sets
  TrainSize = 1000 
  ValidSize = 500
  TrainSet = X[:,1:TrainSize]
  ValidSet = X[:,(TrainSize+1):(TrainSize+ValidSize)]

  # Initialize Model
  crbm = HTreeRBM.HTRBM(VisUnits,HiddenUnits; momentum=0.2, sigma=0.1, sigma_hid=0.1)

  #HTreeRBM.init_vbias!(crbm,TrainSet)
  # Training ...
  HTreeRBM.fit(crbm, TrainSet, n_iter=Epochs; lr=0.1)
  
  return crbm;
end

myrbm = run_artificial_data()
