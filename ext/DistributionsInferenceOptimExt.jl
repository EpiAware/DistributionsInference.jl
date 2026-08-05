# DistributionsInference x Optim: the one optimiser-specific step of a point
# fit. `minimise` is declared with its docstring in `src/optimise.jl`; every
# other step of `optimise_distribution` is package-neutral, so this is the
# whole of what supporting an optimisation package takes.
module DistributionsInferenceOptimExt

using DistributionsInference: DistributionsInference
using Optim: Optim

# `Optim.Options` is a positional argument with method-dependent defaults, so
# it is passed through only when the caller supplies one; the remaining
# keywords go straight to `Optim.optimize`. `autodiff` is the one worth
# setting: `Optim` defaults it to central finite differences, so a
# gradient-based method spends `2n` objective evaluations per gradient unless
# the caller names an `ADTypes` backend.
function DistributionsInference.minimise(objective, init::AbstractVector,
        optimiser::Optim.AbstractOptimizer; options = nothing, kwargs...)
    result = options === nothing ?
             Optim.optimize(objective, init, optimiser; kwargs...) :
             Optim.optimize(objective, init, optimiser, options; kwargs...)
    return Optim.minimizer(result)
end

end # module DistributionsInferenceOptimExt
