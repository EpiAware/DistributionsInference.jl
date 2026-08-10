module DistributionsInferenceDynamicPPLExt

# DistributionsInference x DynamicPPL: `distribution_to_turing(obj, data)`
# builds a DynamicPPL model over a fittable object's estimated rows, wrapping
# the `distribution_to_logdensity` codec (docstring in `src/turing.jl`). Each
# row's dotted `name` becomes the `VarName` `<prefix>.<name>`, sampled from
# its own prior.
#
# `_row_varname` is the naming contract shared with the `VarName`-keyed
# readback in `DistributionsInferenceDynamicPPLFlexiChainsExt` (DI#4, the
# two-extension split); it is declared as a stub in `src/turing.jl` so that
# extension reaches it by dispatch rather than through a sibling extension's
# module.

using DistributionsInference: DistributionsInference, FitLogDensity,
                              distribution_to_logdensity, estimated_rows,
                              reconstruct, extra_logprior,
                              _check_generic_fields, template, observations,
                              flat_priors, TuringEngine
import DistributionsInference: distribution_to_turing, distribution_to_chain,
                               _row_varname
using DynamicPPL: DynamicPPL, @model, NamedDist, VarName, sample
using FlexiChains: FlexiChains, VNChain

# There is no public constructor for a runtime dotted optic, so the two
# primitives `@varname` lowers to are reached through `AbstractPPL` itself:
# `Property{sym}(child)` for a `.sym` access and `Iden()` for the leaf.
const _AbstractPPL = parentmodule(VarName)
const _Property = _AbstractPPL.Property
const _Iden = _AbstractPPL.Iden

# Build a row's `VarName`, e.g. prefix `:d` and `Symbol("onset.shape")` give
# `"d.onset.shape"`. The optic is folded outermost-property-first (`reverse`)
# so the earliest segment renders nearest the prefix.
function _row_varname(prefix::Symbol, name::Symbol)
    segs = Tuple(Symbol(s) for s in split(string(name), "."))
    optic = foldl((acc, s) -> _Property{s}(acc), reverse(segs); init = _Iden())
    return VarName{prefix}(optic)
end

# A row with `prior === nothing` has no distribution to sample it from.
function _validate_turing_rows(obj)
    rows = estimated_rows(obj)
    missing_prior = [row.name for row in rows if row.prior === nothing]
    isempty(missing_prior) || throw(ArgumentError(
        "distribution_to_turing does not support estimated parameter(s) " *
        "$missing_prior with no fixed `~` prior (scored instead through " *
        "`extra_logprior`, an object-dependent prior term whose sampling " *
        "path does not exist yet in DynamicPPL). Sample with " *
        "`distribution_to_logdensity(obj, data)` + LogDensityProblemsAD " *
        "(the LogDensityProblems extension) instead."))
    return rows
end

# `NamedDist` sets the site name to the readback name regardless of the LHS.
# Priors go through `~` and everything else through `@addlogprob!`, so there
# is no double counting and the total equals `logdensity(prob, θ)`. `θ` has an
# abstract element type so AD values flow through `reconstruct` unchanged.
@model function _fit_turing_model(prob::FitLogDensity, vns)
    fp = flat_priors(prob)
    n = length(fp)
    θ = Vector{Real}(undef, n)
    for i in 1:n
        param ~ NamedDist(fp[i], vns[i])
        θ[i] = param
    end
    obj_template = template(prob)
    _check_generic_fields(typeof(obj_template), prob.concrete_fields, θ)
    obj = reconstruct(obj_template, θ)
    DynamicPPL.@addlogprob! extra_logprior(obj_template, obj, θ, prob.extra_state)
    DynamicPPL.@addlogprob! prob.loglik(obj, observations(prob))
    return obj
end

function distribution_to_turing(obj, data;
        prefix::Symbol = :d, loglik = DistributionsInference._default_loglik)
    rows = _validate_turing_rows(obj)
    prob = distribution_to_logdensity(obj, data; loglik = loglik)
    vns = [_row_varname(prefix, row.name) for row in rows]
    return _fit_turing_model(prob, vns)
end

# `TuringEngine` (declared in core, `src/inference_engine.jl`): the engine
# contract's Turing consumer, `distribution_to_turing` + `sample`. Pooling
# `nchains > 1` runs by hand (rather than reaching for `AbstractMCMC`'s
# `MCMCThreads`) keeps this extension triggered by `DynamicPPL` alone.
function DistributionsInference.distribution_to_chain(
        obj, data, engine::TuringEngine;
        loglik = DistributionsInference._default_loglik,
        prefix::Symbol = :d, kwargs...)
    model = distribution_to_turing(obj, data; prefix = prefix, loglik = loglik)
    run_kwargs = merge(engine.kwargs, NamedTuple(kwargs))
    chains = [sample(model, engine.alg, engine.nsamples;
                  chain_type = VNChain, run_kwargs...)
              for _ in 1:engine.nchains]
    return engine.nchains == 1 ? only(chains) : _pool_vnchains(chains)
end

# Pool `n` independent single-chain `VNChain`s into one `n`-chain `VNChain`,
# chain-major — the same pooling convention `draws_to_chain` uses for raw
# draws. Every chain samples the same model, so they share `niters` and the
# same `VarName` parameter set; `only(unique(...))` turns a mismatch (a
# caller-supplied `kwargs` changing `nsamples` mid-run some other way) into a
# named error rather than a silent `hcat` dimension mismatch two frames down.
function _pool_vnchains(chains::AbstractVector)
    niters = only(unique(FlexiChains.niters.(chains)))
    vns = FlexiChains.parameters(first(chains))
    data = Dict{FlexiChains.ParameterOrExtra{<:VarName}, Matrix}()
    for vn in vns
        data[FlexiChains.Parameter(vn)] = hcat((chain[vn] for chain in chains)...)
    end
    return FlexiChains.FlexiChain{VarName}(niters, length(chains), data)
end

end # module DistributionsInferenceDynamicPPLExt
