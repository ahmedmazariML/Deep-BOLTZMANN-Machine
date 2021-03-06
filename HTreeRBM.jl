
module HTreeRBM

export HTRBM,
       BPTree,
       Monitor,
       LogisticReg,
       gibbs,
       sample_hid,
       sample_vis,
       means_vis,
       means_hid,
       get_hid,
       BP_update,
       BP_marg,
       BP_init,
       BP_iter_fix,
       BP_iter_marg,
       BP_compute_marg,
       BP_compute_2pmarg,
       BP_leaves_to_center,
       BP_center_to_leaves,
       fit,
       init_vbias!,
       predict,
       predictfull,
       binarize!,
       components,
       features,
       Monitor,
       binarize,
       binarize!,
       normalize,
       normalize!,
       normalize_samples,
       normalize_samples!,
       ShowMonitor,
       SaveMonitor,
       plot_scores,
       plot_evolution,
       plot_rf,
       plot_chain,
       plot_vbias,
       plot_weightdist,
       plot_hist

include("core.jl")

end
