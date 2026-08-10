# DistributionsInference x AdvancedMH x Bijectors: the engine contract's
# random-walk-Metropolis consumer (`MetropolisEngine`, declared in core at
# `src/inference_engine.jl`).
#
# `AdvancedMH` proposes on whatever scale its `DensityModel` is built over;
# the tutorials' hand-rolled version builds it over the CONSTRAINED scale and
# guards every evaluation by hand (`any(<=(0), x) ? -Inf : ...`), since a
# random-walk step can propose outside a prior's support. Building the model
# over the UNCONSTRAINED scale instead (via `to_constrained`, so this needs
# `Bijectors` loaded as well as `AdvancedMH` — hence the two-package
# extension trigger in `Project.toml`) retires that guard for every row with
# its own prior: every unconstrained point maps to a valid constrained one.
# A row scored through `extra_logprior` instead (no prior of its own) has no
# bijector to build from `to_constrained` either, so an object with such a
# row is still out of reach here — the same row kind `distribution_to_turing`
# refuses, and for the same reason.
module DistributionsInferenceAdvancedMHExt

using DistributionsInference: DistributionsInference, MetropolisEngine,
                              distribution_to_logdensity, template,
                              flat_dimension, to_constrained, draws_to_chain
using AdvancedMH: AdvancedMH, RWMH, DensityModel
using Distributions: MvNormal
using LinearAlgebra: I
using Random: Random

# The default proposal once `dim` (the estimated flat dimension) is known;
# `MetropolisEngine`'s own constructor cannot size this, since it is built
# before `obj`/`data` are in hand.
_default_sampler(dim::Int, proposal_scale::Real) = RWMH(
    MvNormal(zeros(dim), proposal_scale^2 * I))

function DistributionsInference.distribution_to_chain(
        obj, data, engine::MetropolisEngine;
        loglik = DistributionsInference._default_loglik,
        rng = Random.default_rng(), kwargs...)
    prob = distribution_to_logdensity(obj, data; loglik = loglik)
    dim = flat_dimension(template(prob))

    # Nothing estimated: no chain to sample, mirroring
    # `optimise_distribution`'s zero-dimension shortcut. `nsamples` still
    # sets the draw count, so a caller reading `draws_to_chain`'s length gets
    # what they asked for.
    if dim == 0
        draws = [Float64[] for _ in 1:(engine.nsamples - engine.burnin)]
        return draws_to_chain(obj, draws; nchains = 1)
    end

    model = DensityModel() do z
        x, logjac = to_constrained(prob, z)
        DistributionsInference.logdensity(prob, x) + logjac
    end
    sampler = engine.sampler === nothing ?
              _default_sampler(dim, engine.proposal_scale) : engine.sampler
    # `progress = false` is the default here (a chain per `nchains` would
    # otherwise print one bar per chain); a caller's own `progress` in
    # `engine.kwargs`/`kwargs` overrides it.
    run_kwargs = merge((; progress = false), engine.kwargs, NamedTuple(kwargs))

    all_draws = Vector{Vector{Float64}}()
    for _ in 1:engine.nchains
        transitions = AdvancedMH.sample(
            rng, model, sampler, engine.nsamples; run_kwargs...)
        for t in transitions[(engine.burnin + 1):end]
            x, _ = to_constrained(prob, t.params)
            push!(all_draws, x)
        end
    end
    return draws_to_chain(obj, all_draws; nchains = engine.nchains)
end

end # module DistributionsInferenceAdvancedMHExt
