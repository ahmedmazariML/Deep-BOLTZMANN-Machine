workspace()

using MNIST
import HTreeRBM


function run_mnist()
  # Set parameters
  Epochs = 100
  HiddenUnits = 10

  # Get MNIST training data
  X, labels = traindata()
  VisUnits = size(X[:,1],1)
  HTreeRBM.binarize!(X)

  # Split sets
  TrainSize = 10000
  ValidSize = 500
  TrainSet = round(Int,X[:,1:TrainSize])
  ValidSet = X[:,(TrainSize+1):(TrainSize+ValidSize)]
  TrainLabels = round(Int64,labels[1:TrainSize])
  ValidLabels = round(Int64,labels[(TrainSize+1):(TrainSize+ValidSize)])
  labels = round(Int64,labels)
  X = round(Int64,X)

  # Initialize Model
  println("Vis-hidden=",VisUnits,"-",HiddenUnits)
  myLogR = HTreeRBM.LogisticReg(VisUnits,HiddenUnits)

  # Training...
  HTreeRBM.fit(myLogR, TrainSet, TrainLabels, X, labels, TrainSize, n_iter=Epochs; lr=0.1, batch_size=100)
  #HTreeRBM.fit(myLogR, TrainSet, TrainLabels, n_iter=Epochs; lr=1.0, batch_size=1000)
  print("Training done!")
  return myLogR;
end


myLogR = run_mnist()

# Test Predict
x,l = traindata()
HTreeRBM.binarize!(x)
x = round(Int,x)

# Error in training set
p=HTreeRBM.predictfull(myLogR,x[:,1:50000])
errTrain = sum(1-(p.==l[1:50000]))/50000

# Error in test set
p=HTreeRBM.predictfull(myLogR,x[:,50001:60000])
errTest = sum(1-(p.==l[50001:60000]))/10000
