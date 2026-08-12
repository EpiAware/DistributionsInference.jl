module DistributionsInferenceDynamicPPLExt

# DistributionsInference x DynamicPPL: `distribution_to_turing(obj, data)`
# builds a DynamicPPL model over a fittable object's estimated rows, wrapping
# the `distribution_to_logdensity` codec (docstring in `src/turing.jl`). Each
# row's dotted `name` becomes the `VarName` `<prefix>.<name>`, sampled from
# its own prior. `distribution_to_turing(obj, data, sampler, nsamples)` is
# the second form (#94, docstring alongside its definition below): the same
# model, sampled with `sample(...; chain_type = VNChain)`, returned as-is — a
# `VarName`-keyed `FlexiChain`, which `inference_to_distribution`/
# `inference_to_distributions` already read
# (`DistributionsInferenceDynamicPPLFlexiChainsExt`), so no rekeying onto
# `Symbol` is needed here.
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
                              flat_priors
import DistributionsInference: distribution_to_turing, _row_varname
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

@doc "

Sample a fittable object's posterior with `Turing`, and return a `FlexiChain`.

`distribution_to_turing(obj, data, sampler, nsamples; nchains = 1, prefix,
loglik, kwargs...)` drives
[`distribution_to_turing`](@ref)`(obj, data; prefix, loglik)`'s model with
`sample(model, sampler, nsamples; chain_type = FlexiChains.VNChain,
kwargs...)`, so
[`inference_to_distribution`](@ref)/[`inference_to_distributions`](@ref)
read the result exactly as they do a chain built by hand from
`sample(distribution_to_turing(obj, data), ...)`. This is a different
concrete return type from the 2-argument
[`distribution_to_turing`](@ref)`(obj, data)` (a `DynamicPPL` model, unaffected
by this method): one function, two return types picked by arity, the same
pattern [`inference_to_distribution`](@ref) uses for its own two forms.

`nchains > 1` runs that many independent `sample` calls and pools them into
one multi-chain `FlexiChain`, chain-major, the same convention
[`draws_to_chain`](@ref) uses.

This method is available only when `DynamicPPL` is loaded (`Turing`, in
practice, for a useful `sampler` such as `NUTS()`).

# Arguments
- `obj`: the template fittable object, carrying its [`parameter_rows`](@ref).
- `data`: the observed records scored by `loglik`.
- `sampler`: the `Turing`/`DynamicPPL` sampler, e.g. `NUTS()`.
- `nsamples`: the number of samples per chain.

# Keyword Arguments
- `nchains`: the number of independent chains to sample (default `1`).
- `prefix`: the outer submodel variable name the sites are namespaced under
  (default `:d`), matching [`distribution_to_turing`](@ref)`(obj, data)`.
- `loglik`: a reducer `(obj, data) -> Real` scoring `data` against the
  reconstructed object (default: sum of `logpdf(obj, record)`).
- other keywords are forwarded to `sample`, e.g. `progress = false`.

# Examples
```@example
using DistributionsInference, Distributions, DynamicPPL, Turing, Random

struct TuringChainLeaf{S <: Real}
    shape::S
    scale::Float64
end

Distributions.logpdf(d::TuringChainLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::TuringChainLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::TuringChainLeaf, x::AbstractVector)
    return TuringChainLeaf(x[1], d.scale)
end

leaf = TuringChainLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2]

Random.seed!(1)
chain = distribution_to_turing(leaf, data, NUTS(), 200; progress = false)
fitted = inference_to_distribution(leaf, chain, mean)
fitted.scale  # the fixed parameter, untouched
```

# See also
- [`distribution_to_turing`](@ref)`(obj, data)`: the model-building form this
  samples.
- [`distribution_to_advancedmh`](@ref): the sibling sampling verb.
- [`inference_to_distribution`](@ref) / [`inference_to_distributions`](@ref):
  read the returned chain back onto `obj`.
"
function distribution_to_turing(obj, data, sampler, nsamples::Integer;
        nchains::Integer = 1, prefix::Symbol = :d,
        loglik = DistributionsInference._default_loglik, kwargs...)
    model = distribution_to_turing(obj, data; prefix = prefix, loglik = loglik)
    chains = [sample(model, sampler, nsamples; chain_type = VNChain, kwargs...)
              for _ in 1:nchains]
    return nchains == 1 ? only(chains) : _pool_vnchains(chains)
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
