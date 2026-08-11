# DistributionsInference x AdvancedMH x Bijectors: `distribution_to_advancedmh`
# (#94, docstring in `src/advancedmh.jl`), the `AdvancedMH` sampling verb
# alongside `distribution_to_turing`.
#
# `AdvancedMH` proposes on whatever scale its `DensityModel` is built over;
# the getting-started tutorials' hand-rolled version (before this function
# existed) built it over the CONSTRAINED scale and guarded every evaluation by
# hand (`any(<=(0), x) ? -Inf : ...`), since a random-walk step can propose
# outside a prior's support. Building the model over the UNCONSTRAINED scale
# instead (via `to_constrained`, so this needs `Bijectors` loaded as well as
# `AdvancedMH` — hence the two-package extension trigger in `Project.toml`)
# retires that guard for every row with its own prior: every unconstrained
# point maps to a valid constrained one. A row scored through `extra_logprior`
# instead (no prior of its own) has no bijector to build from `to_constrained`
# either, so an object with such a row is still out of reach here — the same
# row kind `distribution_to_turing` refuses, and for the same reason.
module DistributionsInferenceAdvancedMHExt

using DistributionsInference: DistributionsInference, distribution_to_logdensity,
                              to_constrained, draws_to_chain
using AdvancedMH: AdvancedMH, DensityModel
using Random: Random

function DistributionsInference.distribution_to_advancedmh(
        obj, data, sampler, nsamples::Integer;
        nchains::Integer = 1, burnin::Integer = 0,
        loglik = DistributionsInference._default_loglik,
        rng = Random.default_rng(), kwargs...)
    0 <= burnin < nsamples || throw(ArgumentError(
        "distribution_to_advancedmh: burnin must satisfy " *
        "0 <= burnin < nsamples (got burnin=$burnin, nsamples=$nsamples)"))
    prob = distribution_to_logdensity(obj, data; loglik = loglik)

    model = DensityModel() do z
        x, logjac = to_constrained(prob, z)
        DistributionsInference.logdensity(prob, x) + logjac
    end

    # `progress = false` is the default here (a chain per `nchains` would
    # otherwise print one bar per chain); a caller's own `progress` in
    # `kwargs` overrides it.
    run_kwargs = merge((; progress = false), NamedTuple(kwargs))

    all_draws = Vector{Vector{Float64}}()
    for _ in 1:nchains
        transitions = AdvancedMH.sample(
            rng, model, sampler, nsamples; run_kwargs...)
        for t in transitions[(burnin + 1):end]
            x, _ = to_constrained(prob, t.params)
            push!(all_draws, x)
        end
    end
    return draws_to_chain(obj, all_draws; nchains = nchains)
end

end # module DistributionsInferenceAdvancedMHExt
