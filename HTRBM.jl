
using Base.LinAlg.BLAS
using Distributions
using PyCall
using HDF5
using JLD
@pyimport matplotlib.pyplot as plt

import Base.getindex
import StatsBase.fit

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}

abstract AbstractRBM

type HTRBM <: AbstractRBM
  W::Matrix{Float64}
  J::Matrix{Float64}
  vbias::Vector{Float64}
  hbias::Vector{Float64}
  dW::Matrix{Float64}
  dW_prev::Matrix{Float64}
  dJ::Matrix{Float64}
  dJ_prev::Matrix{Float64}
  momentum::Float64
  VisShape::Tuple{Int,Int}
end


function HTRBM(n_vis::Int, n_hid::Int; sigma=0.1, sigma_hid=0.1, momentum=0.0)
  srand(10)
 # b= sqrt(6)/(sqrt(n_hid+n_vis))
  info("=====================================")
  info("RBM parameters")
  info("=====================================")
  info("  + Sigma vis-weights:              $sigma")
  info("  + Sigma hidLayer-weights(unused): $sigma_hid")
  HTRBM(rand(Normal(0, sigma), (n_hid, n_vis)),
    #zeros(n_hid,n_hid),
    #rand(Uniform(-b,b),n_hid,n_vis),
    rand(Normal(0, sigma_hid), (n_hid, n_hid)),
    zeros(n_vis), zeros(n_hid),
    zeros(n_hid, n_vis),
    zeros(n_hid, n_vis),
    zeros(n_hid, n_hid),
    zeros(n_hid, n_hid),
    momentum,
    (n_vis,n_hid))
end


function init_vbias!(rbm::HTRBM,dataset)
  eps = 1e-8

  ProbVis = mean(dataset,2)   # Mean across samples
  ProbVis = max(ProbVis,eps)
  ProbVis = min(ProbVis,1 - eps)

  InitVis = log(ProbVis ./ (1-ProbVis))
  rbm.vbias = vec(InitVis)
end

function gibbs(rbm::HTRBM,bp::BPTree,vis; n_times=1)
  v_pos = vis
  h_pos = sample_hid(rbm,bp,v_pos)


 # v_neg = sample_vis(rbm,hid)
  v_neg = sample_vis(rbm,h_pos)
  h_neg = sample_hid(rbm,bp,v_neg)
 h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/gibbs_sampling/v_neg.h5", "w") do file
    write(file, "a",  v_neg)  # alternatively, say "@write file a"
     end
   h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/gibbs_sampling/h_neg.h5", "w") do file
    write(file, "a",  h_neg)  # alternatively, say "@write file a"
     end
  x=h_neg * v_neg'

  h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/gibbs_sampling/gibbs0.h5", "w") do file
    write(file, "a", x)  # alternatively, say "@write file a"
     end
 println("hello here it is gibbs sampling ")
  for i=1:n_times-1
   v_neg = sample_vis(rbm,h_neg)
   h_neg = sample_hid(rbm,bp,v_neg)
   x=h_neg* v_neg'
  println("allo",i)
  h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/gibbs_sampling/gibbs$i.h5", "w") do file
    write(file, "a", x)  # alternatively, say "@write file a"
    end
  end

  return v_pos,h_pos,v_neg,h_neg

end


function CD(rbm::HTRBM,bp::BPTree,vis; n_times=1)
  v_pos = vis
  #h_pos,h_pos_cov,h_samples = get_hid2(rbm,v_pos) #sample hidden without BPtree
  h_pos,h_pos_cov,h_samples = get_hid(rbm,bp,v_pos)

  # CD1
  v_neg = sample_vis(rbm,h_samples)
  #h_neg,h_neg_cov,h_samples = get_hid2(rbm,v_neg)
 h_neg,h_neg_cov,h_samples = get_hid(rbm,bp,v_neg)
  x=h_neg* v_neg'
  h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/contrastive_divergence/cd0.h5", "w") do file
    write(file, "a", x)  # alternatively, say "@write file a"
     end
#println("coucou ca va  ?")
  # CDn
println("here it is contrastive divergence")
  for i in 1: n_times-1
    v_neg = sample_vis(rbm,h_samples)
    h_neg,h_neg_cov,h_samples = get_hid(rbm,bp,v_neg)
    x=h_neg* v_neg'
  println("allo",i)
  h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/contrastive_divergence/cd$i.h5", "w") do file
    write(file, "a", x)  # alternatively, say "@write file a"
     end
  
  end

 #println("it is the contrastive divegence")


  return v_pos,h_pos,h_pos_cov,v_neg,h_neg,h_neg_cov
end

function get_hid2(rbm::HTRBM,vis)
  means = means_hid(rbm,vis)
  return means,zeros(size(rbm.W,1),size(vis,2)),float(rand(size(means)) .< means)
end

function get_hid(rbm::HTRBM,bp::BPTree,vis)
  h_samples = sample_hid(rbm,bp,vis)
  h_means,h_cov = means_hid(rbm,bp,vis)

  ## println(size(h_samples))
  ## println(size(h_means))
  ## println(size(h_cov))
  
  return h_means,h_cov,h_samples
end

function sample_hid(rbm::HTRBM,bp::BPTree,vis)
  # vis (n_features,n_samples)
  # hid (n_components,n_samples)
  hid = zeros(size(rbm.W)[1],size(vis)[2])
  # w (n_components,n_features)
  # J (n_components,n_components)

  ## # make the tree (should be the same for all samples)
  ## c_tree = caylay_tree(conn,size(hid)[1])
  ## init the Var (same for all samples)
  BP_init(rbm,bp,vis)
  center,_ = BP_leaves_to_center(bp)
  BP_fix_center_to_leaves(bp,center,hid)

  return hid
end

function means_hid(rbm::HTRBM, vis::Mat{Float64})
    p = rbm.W * vis .+ rbm.hbias
    return logistic(p)
end

function means_hid(rbm::HTRBM,bp::BPTree,vis)
  # vis (n_features,n_samples)
  # hid (n_components,n_samples)
  hid = zeros(size(rbm.W)[1],size(vis)[2])
  # w (n_components,n_features)
  # J (n_components,n_components)

  ## # make the tree (should be the same for all samples)
  ## c_tree = caylay_tree(conn,size(hid)[1])
  ## # init the Var (same for all samples)
  BP_init(rbm,bp,vis)
  center,inc = BP_leaves_to_center(bp)
  BP_center_to_leaves(bp,center,inc)
  BP_marg = BP_compute_marg(bp)
  BP_cov,_ = BP_compute_2pmarg(bp)

  # TODO : improve this for
  BP_marg_up = zeros(size(hid))
  for s in 1:size(hid,1)
    BP_marg_up[s,:] = BP_marg[s][:,2]
  end


  return BP_marg_up,BP_cov
end


function means_vis(rbm::HTRBM,hid::Mat{Float64})
  p = rbm.W' * hid .+ rbm.vbias
  return logistic(p)
end



function sample_vis(rbm::HTRBM,hid)
  # vis (n_features,n_samples)
  # hid (n_components,n_samples)
  # w (n_components,n_features)
  means = means_vis(rbm,hid)
  return float(rand(size(means)) .< means)
  ## vis = (p .< (( 1 + tanh(field * 0.5) )/2) )*1.0
end


function update_weights!(rbm, v_pos, h_pos, h_pos_cov, v_neg, h_neg, h_neg_cov, lr)

    ld = 0.001
    # Weights between VISIBLE AND HIDDEN
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, rbm.dW)
    axpy!(-2*ld, rbm.W, rbm.dW)
    # rbm.dW += rbm.momentum * rbm.dW_prev
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)
    # rbm.W += rbm.dW
     
    axpy!(1.0, rbm.dW, rbm.W)
     
    # save current dW
    copy!(rbm.dW_prev, rbm.dW)

    # Weights HIDDEN
    # dJ = (h_pos_cov * h_pos_cov') - (h_neg_cov * h_neg_cov')
    rbm.dJ[1:size(h_neg_cov,1)] = sum(h_neg_cov,2)*lr
    rbm.dJ[1:size(h_pos_cov,1)] = rbm.dJ[1:size(h_neg_cov,1)] - sum(h_pos_cov,2)*lr
    # gemm!('N', 'T', 1.0, h_neg_cov, h_neg_cov, 0.0, rbm.dJ)
    # gemm!('N', 'T', 1.0, h_pos_cov, h_pos_cov, -1.0, rbm.dJ)
    # rbm.dJ += rbm.momentum * rbm.dJ_prev
    axpy!(rbm.momentum, rbm.dJ_prev, rbm.dJ)

    # rbm.J += rbm.dJ
    axpy!(1.0, rbm.dJ, rbm.J)
    # save current dJ
    copy!(rbm.dJ_prev, rbm.dJ)

#h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/v_neg_2.h5", "w") do file
 #   write(file, "a", v_neg)  # alternatively, say "@write file a"
#end
#h5open("/home/anelmad/Desktop/stage-inria/code/HTreeRBM.jl/my_tree/h_neg_2.h5", "w") do file
 #   write(file, "a", h_neg)  # alternatively, say "@write file a"
#end
  
end



function fit_batch!(rbm::HTRBM, vis::Mat{Float64}, lr;
                    n_gibbs=1, conn=3)
    n_hid_nodes = size(rbm.J,1)
    bp = BPTree(n_hid_nodes,conn)
    v_pos,h_pos,v_neg,h_neg=gibbs(rbm,bp,vis; n_times=3)
    v_pos, h_pos, h_pos_cov, v_neg, h_neg, h_neg_cov = CD(rbm,bp,vis;n_times=n_gibbs)
    
    learn_rate = lr/size(h_neg,2)

    update_weights!(rbm, v_pos, h_pos, h_pos_cov, v_neg, h_neg, h_neg_cov, learn_rate)
    axpy!(learn_rate,vec(sum(h_pos-h_neg,2)),rbm.hbias)
    axpy!(learn_rate,vec(sum(v_pos-v_neg,2)),rbm.vbias)
end

function fit(rbm::HTRBM, X::Mat{Float64};
             lr=0.1, n_iter=10, batch_size=10, n_gibbs=1, validation=[],
             monitor_every=1,monitor_vis=false)

    @assert minimum(X) >= 0 && maximum(X) <= 1

    n_valid=0
    nh = size(rbm.W,1)
    nv = size(rbm.W,2)
    N = nh+nv

    n_samples = size(X, 2)
    n_batches = round(Int,ceil(n_samples / batch_size))
    w_buf = zeros(size(rbm.W))

    # Check for the existence of a validation set
    flag_use_validation=false
    if length(validation)!=0
        flag_use_validation=true
        n_valid=size(validation,2)
    end

    # Create the historical monitor
    ProgressMonitor = Monitor(n_iter,monitor_every;monitor_vis=monitor_vis,
                                                   validation=flag_use_validation)

    # Print info to user
    m_ = rbm.momentum
    info("=====================================")
    info("RBM Training")
    info("=====================================")
    info("  + Training Samples:   $n_samples")
    info("  + Features:           $nv")
    info("  + Hidden Units:       $nh")
    info("  + Epochs to run:      $n_iter")
    info("  + Momentum:           $m_")
    info("  + Learning rate:      $lr")
    info("  + Gibbs Steps:        $n_gibbs")
    info("  + Validation Set?:    $flag_use_validation")
    info("  + Validation Samples: $n_valid")
    info("=====================================")

    #UpdateMonitor!(rbm,ProgressMonitor,X,-1;bt=-1.0,validation=validation)
    #ShowMonitor(rbm,ProgressMonitor,-1)

    for itr=1:n_iter
        tic()
        for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            batch = full(batch)
            fit_batch!(rbm, batch, lr,
                       n_gibbs=n_gibbs)
        end
        walltime_µs=(toq()/n_batches/N)*1e6

        UpdateMonitor!(rbm,ProgressMonitor,X,itr;bt=walltime_µs,validation=validation)
        ShowMonitor(rbm,ProgressMonitor,itr)

        ##pseudo_likelihood,pLike_Tree = score_samples(rbm, X)
        ##pseudo_likelihood = mean(pseudo_likelihood)/(nh+nv)
        ##pLike_Tree = mean(pLike_Tree)/(nh+nv)
        ##info("Iteration #$itr, pseudo-likelihood = $pseudo_likelihood")
        ##info("Iteration #$itr, pLike_Tree = $pLike_Tree")
    end

    nw = size(rbm.W[:],1)
    y = collect(1:nw)/nw
    writedlm("coupW.d",[sort(rbm.W[:]) y],'\t')


    nw = size(rbm.J[:],1)
    y = collect(1:nw)/nw
    writedlm("coupJ.d",[sort(rbm.J[:]) y],'\t')
    return rbm,ProgressMonitor
end

function save_parameters(rbm::HTRBM, mon, rep, tag)
  filename=  string(rep,"data_",tag,".h5")
  h5write(filename, "HTRBM/MonPL",mon.PseudoLikelihood)
  h5write(filename, "HTRBM/MonPLT",mon.PseudoLikelihood_Tree)
  h5write(filename, "HTRBM/W",rbm.W)
  h5write(filename, "HTRBM/J",rbm.J)
  h5write(filename, "HTRBM/hb",rbm.hbias)
  h5write(filename, "HTRBM/vb",rbm.vbias)
end

function score_samples(rbm::HTRBM, vis::Mat{Float64}; sample_size=10000)
    sample_size = min(sample_size,size(vis,2))
    if issparse(vis)
        # sparse matrices may be infeasible for this operation
        # so using only little sample
        cols = sample(1:size(vis, 2), sample_size)
        vis = full(vis[:, cols])
    end
    n_feat, n_samples = size(vis)
    vis_corrupted = copy(vis)
    # shuffle the data
    idxs = rand(1:n_feat, n_samples)
    for (i, j) in zip(idxs, 1:n_samples)
        vis_corrupted[i, j] = 1 - vis_corrupted[i, j]
    end
    fe = free_energy(rbm, vis)
    fe_corrupted = free_energy(rbm, vis_corrupted)

    n_hid_nodes = size(rbm.W,1)
    conn = 3
    mybp  = BPTree(n_hid_nodes,conn)
    BP_init(rbm,mybp,vis)
    center,inc = BP_leaves_to_center(mybp)
    BP_center_to_leaves(mybp,center,inc)
    fnrg = BP_compute_fnrg(mybp)' - sum(vis .* rbm.vbias, 1)

    BP_init(rbm,mybp,vis_corrupted)
    center,inc = BP_leaves_to_center(mybp)
    BP_center_to_leaves(mybp,center,inc)
    fnrg_corr = BP_compute_fnrg(mybp)' - sum(vis_corrupted .* rbm.vbias, 1)

    pLike_Ind  = n_feat * log(logistic(fe_corrupted - fe))
    pLike_Tree = n_feat * log(logistic(fnrg_corr - fnrg))
    return pLike_Ind,pLike_Tree
end


function recon_error(rbm::HTRBM, vis::Mat{Float64})

    n_hid_nodes = size(rbm.W,1)
    conn = 3
    mybp  = BPTree(n_hid_nodes,conn)
    BP_init(rbm,mybp,vis)
    ##center,inc = BP_leaves_to_center(mybp)
    ##BP_center_to_leaves(mybp,center,inc)

    # Fully forward MF operation to get back to visible samples
    h_means,_ = means_hid(rbm,mybp,vis)
    vis_rec = means_vis(rbm,h_means)
    # Get the total error over the whole tested visible set,
    # here, as MSE
    dif = vis_rec - vis
    mse = mean(dif.*dif)
    return mse
end

function free_energy(rbm::HTRBM, vis::Mat{Float64})
    vb = sum(vis .* rbm.vbias, 1)
    Wx_b_log = sum(log(1 + exp(rbm.W * vis .+ rbm.hbias)), 1)
    return - vb - Wx_b_log
end

function generate_hidden(rbm::HTRBM, X::Mat{Float64};conn=3)
  # X (n_features,n_samples)

  n_hid_nodes = size(rbm.J,1)
  bp = BPTree(n_hid_nodes,conn)

  return sample_hid(rbm,bp,X)
end

function GetBPStates(rbm::HTRBM, vis;conn=3)
  n_hid_nodes = size(rbm.J,1)
  bp = BPTree(n_hid_nodes,conn)
  hid = zeros(size(rbm.W)[1],size(vis)[2])
  # w (n_components,n_features)
  # J (n_components,n_components)

  ## # make the tree (should be the same for all samples)
  ## c_tree = caylay_tree(conn,size(hid)[1])
  ## # init the Var (same for all samples)
  BP_init(rbm,bp,vis)
  center,inc = BP_leaves_to_center(bp)
  BP_center_to_leaves(bp,center,inc)
  BP_marg = BP_compute_marg(bp)
  BP_cov = BP_compute_2pmarg_full(bp)

  deg = zeros(0)
  for i in vertices(bp.c_gr)
    append!(deg,[in_degree(i,bp.c_gr)])
  end

  return BP_marg,BP_cov,deg
end


function BP_init(rbm::HTRBM, bp::BPTree, vis)
  # compute the local bias : sum of the site bias
  # + effect of the visible layer

  # vis (n_features,n_samples)
  # w (n_components,n_features)
  # J (n_components,n_components)

  resize!(bp.BPmsg,2*num_edges(bp.gr))
  resize!(bp.match_edges,2*num_edges(bp.gr))
  bp.c_gr = copy_gr(bp.gr)

  local_b = rbm.W*vis .+ rbm.hbias
  resize!(bp.Mh,2*num_edges(bp.gr))
  ## println(size(rbm.J))
  ## println(size(bp.Mh))
  for e in edges(bp.gr)
    i_e = edge_index(e,bp.gr)
    Jss = rbm.J[i_e]
    bp.Mh[i_e] = [ 1 1 ; 1 exp(Jss) ]
  end

  resize!(bp.Vb,num_vertices(bp.gr))
  ## println(size(bp.Vb))
  ## println(size(local_b))
  for n in vertices(bp.gr)
    bp.Vb[n] = [exp(-local_b[n,:]*0);exp(local_b[n,:])]'
  end
end
