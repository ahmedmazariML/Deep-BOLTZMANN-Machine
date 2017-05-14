using StatsFuns


#typealias Mat{T} AbstractArray{T, 2}
#typealias Vec{T} AbstractArray{T, 1}

abstract LOGREG

type LogisticReg <: LOGREG
  W::Matrix{Float64}
  vbias::Vector{Float64}
  dW::Matrix{Float64}
  dW_prev::Matrix{Float64}
  momentum::Float64
end


function LogisticReg(n_vis::Int, n_hid::Int; sigma=0.001, momentum=0.1)
  LogisticReg(rand(Normal(0, sigma), (n_hid, n_vis)),
    zeros(n_hid),
    zeros(n_hid, n_vis),
    zeros(n_hid, n_vis),
    momentum)
end

function LabelsBin(Y::Vec{Int}, n_class)
  LabBin = zeros(Int,size(Y,1),n_class)
  for s=1:size(Y,1)
    LabBin[s,Y[s]+1] = 1
  end
  return LabBin
end

function UpdateWeights(LR::LogisticReg, grad_w, grad_b, lr)
  #gemm!('N','',lr,)
end

function multinomial_loss(LR::LogisticReg, X::Mat{Int}, y::Mat{Int})
  # X is (n_feat, n_samples)
  # W is (n_hid, n_feat)
  # y is -> give the label!

  p = LR.W * X  .+ LR.vbias # p -> (n_hid,n_samples)
  p = p .- log( sum(exp(p),1) ) # log of the loss
  loss = -sum( p .* y' )
  return loss,p
end

function multinomial_loss_grad(LR::LogisticReg, X::Mat{Int}, y::Mat{Int})
  # X is (n_feat, n_samples)
  # y = LabelsBin(Y,size(LR.W,2)) is ()

  loss,p = multinomial_loss(LR,X,y) # p is (n_hid,n_samples)
  expp = exp(p)
  diff = -(exp(p) - y') # diff is (n_hid,n_samples)
  ### diff = (p .* y') # diff is (n_hid,n_samples)
  grad_w = diff*X' ./ size(X,2) # grad_w is (n_hid,n_feat)
  grad_bias = sum(diff,2) ./ size(X,2) # grad_bias is (n_hid)

  return loss,grad_w,grad_bias,p
end

function fit(LR::LogisticReg, X::Mat{Int}, Y::Vec{Int}; lr=0.1, n_iter=20, batch_size=20,
             validation=[])

  @assert minimum(X) >= 0 && maximum(X) <= 1

  n_valid=0
  nh = size(LR.W,1)
  nv = size(LR.W,2)
  N = nh+nv

  n_samples = size(X,2)
  n_batches = round(Int,ceil(n_samples / batch_size))

  # Check for the existence of a validation set
  flag_use_validation=false
  if length(validation)!=0
      flag_use_validation=true
      n_valid=size(validation,2)
  end

  y = LabelsBin(Y,size(LR.W,1))

  for itr=1:n_iter
    tic()
    loss,p = multinomial_loss(LR,X,y)
    p_train = predictfull(LR,X)
    err_train = sum(1-(p_train.==Y))/n_samples
    lr *= 0.999
    println(itr," ",loss," ",err_train)
    for i=1:n_batches
        batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
        batch = full(batch)

        batch_lab = Y[((i-1)*batch_size + 1):min(i*batch_size, end)]
        batch_lab = full(batch_lab)

        fit_batch!(LR, batch, batch_lab, lr)
    end

    walltime_µs=(toq()/n_batches/N)*1e6
  end
end

function fit(LR::LogisticReg, X::Mat{Int}, Y::Vec{Int}, X_test::Mat{Int64}, Y_test::Vec{Int}, size_test; lr=0.1, n_iter=20, batch_size=20,
             validation=[])

  @assert minimum(X) >= 0 && maximum(X) <= 1

  n_valid=0
  nh = size(LR.W,1)
  nv = size(LR.W,2)
  N = nh+nv

  n_samples = size(X,2)
  n_batches = round(Int,ceil(n_samples / batch_size))

  # Check for the existence of a validation set
  flag_use_validation=false
  if length(validation)!=0
      flag_use_validation=true
      n_valid=size(validation,2)
  end

  y = LabelsBin(Y,size(LR.W,1))

  for itr=1:n_iter
    tic()
    loss,p = multinomial_loss(LR,X,y)
    lr *= 0.999
    p_test = predictfull(LR,X_test[:,n_samples+1:n_samples+1+size_test])
    p_train = predictfull(LR,X_test[:,1:n_samples])
    err_test = sum(1-(p_test.==Y_test[n_samples+1:n_samples+1+size_test]))/size_test
    err_train = sum(1-(p_train.==Y_test[1:n_samples]))/n_samples
    println(itr," ",loss," ",err_train," ",err_test)

    for i=1:n_batches
        batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
        batch = full(batch)

        batch_lab = Y[((i-1)*batch_size + 1):min(i*batch_size, end)]
        batch_lab = full(batch_lab)

        fit_batch!(LR, batch, batch_lab, lr)
    end

    walltime_µs=(toq()/n_batches/N)*1e6
  end
end

function fit_batch!(LR::LogisticReg, vis, labs, lr)
  # vis is (n_samples, n_features)
  # labs is (n_samples)

  y = LabelsBin(labs,size(LR.W,1))
  loss,grad_w,grad_bias,p = multinomial_loss_grad(LR,vis,y)

  LR.dW = lr*grad_w
  Base.axpy!(LR.momentum, LR.dW_prev, LR.dW)
  Base.axpy!(1.0, LR.dW, LR.W)
  copy!(LR.dW_prev, LR.dW)
  Base.axpy!(lr,grad_bias,LR.vbias)

  # UpdateWeights(LR,grad_w,grad_bias)
end

function predict(LR::LogisticReg, X::Vec{Int})
  # X is (n_feat, n_samples)
  # W is (n_hid, n_feat)
  p = exp(LR.W * X .+ LR.vbias)
  return p./sum(p,1)
end

function predictfull(LR::LogisticReg, X::Mat{Int})
  res = Int[]
  for i=1:size(X,2)
    push!(res,indmax(predict(LR,X[:,i]))-1)
  end
  return res
end
