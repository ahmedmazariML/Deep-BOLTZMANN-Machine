using Images
# using Colors
# using ImageView
# using Gadfly
using PyCall
@pyimport matplotlib.pyplot as plt

function chart_weights(W, imsize; padding=0, annotation="", filename="", noshow=false, ordering=true)
    h, w = imsize
    n = size(W, 1)
    rows = round(Int,floor(sqrt(n)))
    cols = round(Int,ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))

    # Sort receptive fields by energy
    if ordering
        p = sum(W.^2,2)
        p = sortperm(vec(p);rev=true)
        W = W[p,:]
    end

    for i=1:n
        wt = W[i, :]
        wim = reshape(wt, imsize)
        r = div(i - 1, cols) + 1
        c = rem(i - 1, cols) + 1
        dat[(r-1)*(h+padding)+halfpad+1 : r*(h+padding)-halfpad,
            (c-1)*(w+padding)+halfpad+1 : c*(w+padding)-halfpad] = wim
    end

    normalize!(dat)

    # Write to file
    if length(filename) > 0
        Images.imwrite(dat,filename,quality=100)
    end

    return dat
end


function plot_scores(mon::Monitor)
    ax_pl = plt.gca()
    ax_re = ax_pl[:twinx]()

    hpl = ax_pl[:plot](mon.Epochs,mon.PseudoLikelihood,"b^-",label="Pseudo-Likelihood")
    if mon.UseValidation
        hvpl = ax_pl[:plot](mon.Epochs,mon.ValidationPseudoLikelihood,"b^:",label="Pseudo-Likelihood (Validation)")
    end
    ax_pl[:set_ylabel]("Normalized Likelihood")
    ax_pl[:set_ylim]((-0.3,0.0))

    hre = ax_re[:plot](mon.Epochs,mon.ReconError,"-*r",label="Recon. Error")
    if mon.UseValidation
        hvre = ax_re[:plot](mon.Epochs,mon.ValidationReconError,":*r",label="Recon. Error (Validation)")
    end
    ax_re[:set_ylabel]("Value")
    ax_re[:set_yscale]("log")

    plt.title("Scoring")
    plt.xlabel("Training Epoch")
    plt.xlim((1,mon.Epochs[mon.LastIndex]))
    plt.grid("on")
    if mon.UseValidation
        plt.legend(handles=[hpl;hvpl;hre;hvre],loc=4)
    else
        plt.legend(handles=[hpl;hre],loc=4)
    end
end

function plot_evolution(mon::Monitor)
    hbt = plt.plot(mon.Epochs,mon.BatchTime_µs,"-k*",label="Norm. Batch time (µs)")

    plt.legend(handles=hbt,loc=1)
    plt.title("Evolution")
    plt.xlabel("Training Epoch")
    plt.xlim((1,mon.Epochs[mon.LastIndex]))
    plt.grid("on")
end

function plot_rf(rbm::HTRBM)
    # TODO: Implement RF display in the case of 1D signals
    rf = chart_weights(rbm.W,rbm.VisShape; padding=0,noshow=true)
    plt.imshow(rf;interpolation="Nearest")
    plt.title("Receptive Fields")
    plt.gray()
end

function plot_chain(rbm::HTRBM)
    # TODO: Implement Chain display in the case of 1D signals
    pc = chart_weights(rbm.persistent_chain',rbm.VisShape; padding=0,noshow=true,ordering=false)
    plt.imshow(pc;interpolation="Nearest")
    plt.title("Visible Chain")
    plt.gray()
end

function plot_vbias(rbm::HTRBM)
    vectorMode = minimum(rbm.VisShape)==1 ? true : false

    if vectorMode
        plt.plot(rbm.vbias)
        plt.grid("on")
    else
        plt.imshow(reshape(rbm.vbias,rbm.VisShape);interpolation="Nearest")
    end
    plt.title("Visible Biasing")
    plt.gray()
end

function plot_vbiasdist(rbm::HTRBM)
    plt.hist(vec(rbm.vbias);bins=100)
    plt.title("VisBias Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequeny")
end

function plot_hbiasdist(rbm::HTRBM)
    plt.hist(vec(rbm.hbias);bins=100)
    plt.title("HidBias Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequeny")
end

function plot_hidweightdist(rbm::HTRBM)
    plt.hist(vec(rbm.J[1:size(rbm.W,1)]);bins=100)
    plt.title("Hidden Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequeny")
end

function plot_weightdist(rbm::HTRBM)
    plt.hist(vec(rbm.W);bins=100)
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequeny")
end

function figure_refresh(figureHandle)
    figureHandle[:canvas][:draw]()
    plt.show(block=false)
    plt.pause(0.0001)
end


function SaveMonitor(rbm::HTRBM,mon::Monitor,filename::AbstractString)
    savefig = plt.figure(5;figsize=(11,15))
    # Show Per-Epoch Progres
    savefig[:add_subplot](321)
        plot_scores(mon)

    savefig[:add_subplot](322)
        plot_evolution(mon)

    # Show Receptive fields
    savefig[:add_subplot](323)
        plot_rf(rbm)

    # Show the Visible chains/fantasy particle
    savefig[:add_subplot](324)
        plot_chain(rbm)

    # Show the current visible biasing
    savefig[:add_subplot](325)
        plot_vbias(rbm)

    # Show the distribution of weight values
    savefig[:add_subplot](326)
        plot_weightdist(rbm)

    plt.savefig(filename;transparent=true,format="pdf",papertype="a4",frameon=true,dpi=300)
    plt.close()
end

function plot_hist(rbm::HTRBM)
  fig = plt.figure(1;figsize=(6,8))
  fig[:add_subplot](221)
    plot_weightdist(rbm)

  fig[:add_subplot](222)
    plot_hidweightdist(rbm)

  fig[:add_subplot](223)
    plot_vbiasdist(rbm)

  fig[:add_subplot](224)
    plot_hbiasdist(rbm)

  figure_refresh(fig)
  plt.show()
end

function ShowMonitor(rbm::HTRBM,mon::Monitor,itr::Int;filename=[])
    fig = mon.FigureHandle

    if mon.MonitorVisual && itr%mon.MonitorEvery==0
        # Wipe out the figure
        fig[:clf]()

        # Show Per-Epoch Progres
        fig[:add_subplot](321)
            plot_scores(mon)

        fig[:add_subplot](322)
            plot_evolution(mon)

        # Show Receptive fields
        fig[:add_subplot](323)
            plot_rf(rbm)

        # Show the Visible chains/fantasy particle
        fig[:add_subplot](324)
            plot_chain(rbm)

        # Show the current visible biasing
        fig[:add_subplot](325)
            plot_vbias(rbm)

        # Show the distribution of weight values
        fig[:add_subplot](326)
            plot_weightdist(rbm)

        figure_refresh(fig)
    end

    if (mon.MonitorText && itr%mon.MonitorEvery==0) || itr==-1
        li = mon.LastIndex
        ce = mon.Epochs[li]
        if mon.UseValidation
            @printf("[Epoch %04d] Train(pl : %0.3f), Valid(pl : %0.3f) [%0.3f µsec/batch/unit]\n",ce,
                                                                                                   mon.PseudoLikelihood[li],
                                                                                                   mon.ValidationPseudoLikelihood[li],
                                                                                                   mon.BatchTime_µs[li])
        else
            @printf("[Epoch %04d] Train(pl : %0.3f *** pl_tree : %0.3f) -- Recon err %0.3f :   [%0.3f µsec/batch]\n",ce,
                                                                           mon.PseudoLikelihood[li],
                                                                           mon.PseudoLikelihood_Tree[li],
                                                                           log10(mon.ReconError[li]),
                                                                           mon.BatchTime_µs[li])
        end
	@printf("Mean and std : %0.3f %0.3f\n",mean(rbm.W[:]),std(rbm.W[:]))
    end
end
