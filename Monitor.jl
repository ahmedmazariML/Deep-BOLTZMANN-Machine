abstract AbstractMonitor

@runonce type Monitor <: AbstractMonitor
    LastIndex::Int
    UseValidation::Bool
    MonitorEvery::Int
    MonitorVisual::Bool
    MonitorText::Bool
    Epochs::Vector{Float64}
    LearnRate::Vector{Float64}
    Momentum::Vector{Float64}
    PseudoLikelihood::Vector{Float64}
    ValidationPseudoLikelihood::Vector{Float64}
    PseudoLikelihood_Tree::Vector{Float64}
    ValidationPseudoLikelihood_Tree::Vector{Float64}
    ReconError::Vector{Float64}
    ValidationReconError::Vector{Float64}
    BatchTime_µs::Vector{Float64}
    FigureHandle
end


function Monitor(n_iter,monitor_every;monitor_vis=false,monitor_txt=true,validation=false)
    len = convert(Int,floor(n_iter/monitor_every))
    blank_vector1 = vec(fill!(Array(Float64,len,1),convert(Float64,NaN)))
    blank_vector2 = copy(blank_vector1)
    blank_vector3 = copy(blank_vector1)
    blank_vector4 = copy(blank_vector1)
    blank_vector5 = copy(blank_vector1)
    blank_vector6 = copy(blank_vector1)
    blank_vector7 = copy(blank_vector1)
    blank_vector8 = copy(blank_vector1)
    blank_vector9 = copy(blank_vector1)
    blank_vector10 = copy(blank_vector1)

    if monitor_vis
        fh = plt.figure(1;figsize=(11,15))
    else
        fh = NaN
    end

    Monitor(0,                   # Last Index
            validation,          # Flag for validation set
            monitor_every,       # When to display
            monitor_vis,         # Monitor visal display flag
            monitor_txt,         # Monitor text display flag
            blank_vector1,       # Epochs (for x-axes)
            blank_vector2,       # Learn Rate
            blank_vector3,       # Momentum
            blank_vector4,       # Pseudo-Likelihood
            blank_vector5,       # Validation Pseudo-Likelihood
            blank_vector6,       # Pseudo-Likelihood Tree
            blank_vector7,       # Validation Pseudo-Likelihood Tree
            blank_vector8,       # ReconError
            blank_vector9,       # ValidationReconError
            blank_vector10,       # BatchTime_µs
            fh)                  # Monitor Figure Handle
end

function UpdateMonitor!(rbm::HTRBM,mon::Monitor,dataset::Mat{Float64},itr::Int;validation=[],bt=NaN,lr=NaN,mo=NaN)
    nh = size(rbm.W,1)
    nv = size(rbm.W,2)
    N = nh + nv
    nsamps = min(size(dataset,2),5000)      # Set maximum number of samples to test as 5000

    if itr%mon.MonitorEvery==0 || itr==-1
        if mon.UseValidation
            vpl = mean(score_samples(rbm, validation))/N
            vre = recon_error(rbm,validation)/N
        else
            vpl = NaN
            vre = NaN
        end
        pl,pl_tree = score_samples(rbm, dataset[:,1:nsamps])
        pl = mean(pl)/N
        pl_tree = mean(pl_tree)/N
        re = recon_error(rbm,dataset[:,1:nsamps])/N

        mon.LastIndex+=1
        li = mon.LastIndex

        mon.PseudoLikelihood[li] = pl
        mon.PseudoLikelihood_Tree[li] = pl_tree
        mon.ReconError[li] = re
        mon.ValidationPseudoLikelihood[li] = vpl
        mon.ValidationReconError[li] = vre
        mon.Epochs[li] = itr
        mon.Momentum[li] = mo
        mon.LearnRate[li] = lr
        mon.BatchTime_µs[li] = bt
    end
end
