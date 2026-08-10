# The inference-engine contract (#94): `distribution_to_chain(obj, data,
# engine)` dispatches on `engine`'s own concrete type and returns a
# `FlexiChains.FlexiChain` keyed by the estimated rows' dotted names — the
# same keying `draws_to_chain` produces — so the existing
# `inference_to_distribution`/`inference_to_distributions` readback consumes
# the result with no extra work.
#
# No `AbstractInferenceEngine`: an engine duck-types on its own concrete type,
# which is deliberately not shared with any sibling engine. A hierarchy before
# a second engine exists is structure invented for one instance.
#
# `TuringEngine`/`MetropolisEngine` are the two shipped engines. Their structs
# are declared here, in core, with no PPL dependency of their own (every field
# is generic), so the exported name resolves whether or not the corresponding
# extension has loaded — an extension can only ADD METHODS to an existing
# name, not inject a brand-new binding into this module's namespace. Their
# `distribution_to_chain` methods live in `DistributionsInferenceDynamicPPLExt`
# and `DistributionsInferenceAdvancedMHExt` respectively, built only from this
# package's public surface (`template`, `observations`, `flat_priors`,
# `estimated_rows`, `to_constrained`/`to_unconstrained`, `draws_to_chain`).
#
# `PigeonsEngine` was the first choice for the second engine (#94): Pigeons
# consumes a target built entirely from the public surface, with no internal
# reached for, and its draws key into a `FlexiChain` exactly like any other
# raw-draws sampler's (see `draws_to_chain`). It does not ship here because of
# a dependency conflict outside this package's control, not a contract gap:
# Pigeons 0.4's own `[compat]` caps `DynamicPPL` at 0.40 (confirmed unchanged
# on its `main` branch too), while this package's `Turing` floor (0.45, 0.46)
# needs `DynamicPPL` >= 0.41.3, so the two cannot resolve in one environment
# today. `MetropolisEngine` around `AdvancedMH` is the documented fallback for
# exactly this situation, and additionally retires most of the hand-written
# `-Inf` support guard the README and two tutorials used to carry (every
# occurrence except one: a centred-pooled `ComposedDistributions` tree's rows
# have no per-row prior to build an unconstrained transform from, so that one
# spot keeps sampling — and guarding — on the constrained scale by hand,
# which is also the point that tutorial section makes).

@doc "

Sample a fittable object's posterior with an inference engine.

`distribution_to_chain(obj, data, engine; loglik, kwargs...)` dispatches on
`engine`'s own concrete type and returns a `FlexiChains.FlexiChain` keyed by
[`estimated_rows`](@ref)`(obj)`'s dotted names — the naming
[`draws_to_chain`](@ref) produces and
[`inference_to_distribution`](@ref)/[`inference_to_distributions`](@ref)
already consume, so the result reads back with no extra work:

```julia
chain = distribution_to_chain(delay, data, TuringEngine(NUTS(); nsamples = 1000))
predictive = inference_to_distribution(delay, chain)
draws = inference_to_distributions(delay, chain; draws = 500)
```

An engine builds the problem, drives it however it likes, and hands the
result back as a chain. This package ships two: [`TuringEngine`](@ref) (the
`DynamicPPL` extension) and [`MetropolisEngine`](@ref) (the `AdvancedMH` +
`Bijectors` extension). A third-party engine plugs in with one method on its
own concrete type — see \"Writing an engine\" below — with no supertype to
subtype and no registration step.

# Arguments
- `obj`: the template fittable object, carrying its [`parameter_rows`](@ref).
- `data`: the observed records.
- `engine`: the engine to sample with, e.g. [`TuringEngine`](@ref)`(NUTS();
  nsamples = 1000)` or [`MetropolisEngine`](@ref)`(; nsamples = 2000)`. Its
  concrete type selects the method.

# Keyword Arguments
- `loglik`: a reducer `(obj, data) -> Real` scoring `data` against the
  reconstructed object (default: sum of `logpdf(obj, record)`), the same
  default [`distribution_to_logdensity`](@ref) uses.
- other keywords are forwarded to the engine's own method.

# Writing an engine

An engine is any type; `distribution_to_chain` never inspects it beyond
dispatch. A method built entirely from this package's public surface can:

- assemble [`distribution_to_logdensity`](@ref)`(obj, data; loglik)` — a
  `LogDensityProblems`-conforming object any such sampler can drive directly;
- read [`template`](@ref), [`observations`](@ref) and [`flat_priors`](@ref)
  off it instead of a struct field;
- read [`estimated_rows`](@ref)`(obj)` for the flat vector's names and
  starting values, and [`to_constrained`](@ref)/[`to_unconstrained`](@ref) for
  an unconstrained sampling scale (needs `Bijectors` loaded);
- key its raw draws into a chain with [`draws_to_chain`](@ref)`(obj, draws;
  nchains)` when its sampler does not already hand back a `FlexiChain`
  (`Turing`'s `sample(...; chain_type = FlexiChains.VNChain)` does; a bare
  `LogDensityProblems` sampler's draws do not).

# Examples
```@example
using DistributionsInference, Distributions, DynamicPPL, Turing, Random

struct EngineLeaf{S <: Real}
    shape::S
    scale::Float64
end

Distributions.logpdf(d::EngineLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::EngineLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::EngineLeaf, x::AbstractVector)
    return EngineLeaf(x[1], d.scale)
end

leaf = EngineLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2]

Random.seed!(1)
chain = distribution_to_chain(leaf, data, TuringEngine(NUTS(); nsamples = 200))
fitted = point_estimate(leaf, chain)
fitted.scale  # the fixed parameter, untouched
```

# See also
- [`TuringEngine`](@ref), [`MetropolisEngine`](@ref): the two shipped engines.
- [`inference_to_distribution`](@ref), [`inference_to_distributions`](@ref):
  the conversion verbs that consume the returned chain.
- [`draws_to_chain`](@ref): keys raw draws into the same chain shape, for an
  engine whose sampler has no chain of its own.
"
function distribution_to_chain end

@doc "

An inference engine that samples through `DynamicPPL`/`Turing`.

`TuringEngine(alg; nsamples, nchains = 1, kwargs...)` drives
[`distribution_to_turing`](@ref)`(obj, data; loglik)` with `sample(model, alg,
nsamples; chain_type = FlexiChains.VNChain, kwargs...)`, so
[`distribution_to_chain`](@ref)`(obj, data, TuringEngine(alg; nsamples))`
makes the `Turing` path a *consumer* of the engine contract rather than a
separate entry point beside it — [`distribution_to_turing`](@ref) itself is
unaffected and stays the way to build the model by hand.

`nchains > 1` samples that many independent chains (each its own `sample`
call) and pools them into one multi-chain `FlexiChain`, chain-major, the same
convention [`draws_to_chain`](@ref) uses.

This struct carries no `DynamicPPL` dependency itself (`alg` is opaque here);
its [`distribution_to_chain`](@ref) method has no body until `DynamicPPL` is
loaded, and lives in the `DistributionsInferenceDynamicPPLExt` package
extension.

# Arguments
- `alg`: the `Turing`/`DynamicPPL` sampler, e.g. `NUTS()`.

# Keyword Arguments
- `nsamples`: the number of samples per chain.
- `nchains`: the number of independent chains to sample (default `1`).
- other keywords are forwarded to `sample`, e.g. `progress = false`.

# Examples
```julia
using DistributionsInference, DynamicPPL, Turing

chain = distribution_to_chain(delay, data, TuringEngine(NUTS(); nsamples = 1000))
```

# See also
- [`distribution_to_chain`](@ref): the contract this satisfies.
- [`distribution_to_turing`](@ref): the model this wraps.
- [`MetropolisEngine`](@ref): the sibling engine.
"
struct TuringEngine{A, K <: NamedTuple}
    alg::A
    nsamples::Int
    nchains::Int
    kwargs::K
end

function TuringEngine(alg; nsamples::Integer, nchains::Integer = 1, kwargs...)
    nsamples > 0 || throw(ArgumentError(
        "TuringEngine: nsamples must be positive, got $nsamples"))
    nchains > 0 || throw(ArgumentError(
        "TuringEngine: nchains must be positive, got $nchains"))
    return TuringEngine(alg, Int(nsamples), Int(nchains), NamedTuple(kwargs))
end

# Kept strictly less specific than the extension's exact-arity method (a
# trailing `Vararg`), so both coexist and the extension's takes precedence
# once `DynamicPPL` is loaded — the same pattern `distribution_to_turing`
# uses for the same reason.
function distribution_to_chain(obj, data, engine::TuringEngine, rest...; kwargs...)
    return _extension_required(:distribution_to_chain, "DynamicPPL",
        "DistributionsInferenceDynamicPPLExt")
end

@doc "

An inference engine that samples through `AdvancedMH.jl`'s random-walk
Metropolis.

`MetropolisEngine(sampler = nothing; nsamples, burnin = 0, nchains = 1,
proposal_scale = 0.1, kwargs...)` drives
[`distribution_to_logdensity`](@ref)`(obj, data; loglik)` on the unconstrained
scale (via [`to_constrained`](@ref), so it needs `Bijectors` loaded as well as
`AdvancedMH`) with `AdvancedMH.sample`, and maps the draws back to the
constrained scale with [`draws_to_chain`](@ref).

Sampling on the unconstrained scale is what retires the hand-written `-Inf`
support guard the README and getting-started tutorials used to carry: a
random-walk proposal on the constrained scale can land outside a prior's
support (a negative rate, say), which the tutorials' `AdvancedMH.DensityModel`
guarded by hand (`any(<=(0), x) ? -Inf : ...`); every point on the
unconstrained scale maps to a valid constrained one, so no such guard is
needed here. This needs every estimated row to carry its own prior, exactly
[`to_constrained`](@ref)'s own requirement — a row scored instead through
[`extra_logprior`](@ref) (an object-dependent prior, e.g. a centred-pooled
`ComposedDistributions` tree) has no bijector to build, the same row kind
[`TuringEngine`](@ref)/[`distribution_to_turing`](@ref) also refuses. Such an
object still fits through [`distribution_to_logdensity`](@ref) driven by
hand, on the constrained scale, where the hand-written guard is still needed.

`sampler` is an `AdvancedMH.MHSampler` (e.g. `RWMH(...)`); the default
(`nothing`) builds `RWMH(MvNormal(zeros(dim), proposal_scale^2 * I))` once
`dim` (the estimated flat dimension) is known, so a caller need not size a
proposal by hand for the common case. `burnin` drops that many draws from the
start of each chain before pooling. `nchains > 1` runs that many independent
`sample` calls and pools them into one multi-chain `FlexiChain`, chain-major.

This struct carries no `AdvancedMH` dependency itself; its
[`distribution_to_chain`](@ref) method has no body until `AdvancedMH` and
`Bijectors` are both loaded, and lives in the
`DistributionsInferenceAdvancedMHExt` package extension.

# Arguments
- `sampler`: an `AdvancedMH.MHSampler`, or `nothing` for the default
  `RWMH(MvNormal(zeros(dim), proposal_scale^2 * I))`.

# Keyword Arguments
- `nsamples`: the number of samples per chain (including any `burnin`).
- `burnin`: draws dropped from the start of each chain before pooling
  (default `0`).
- `nchains`: the number of independent chains to sample (default `1`).
- `proposal_scale`: the default proposal's per-coordinate step size (default
  `0.1`); ignored when `sampler` is given explicitly.
- other keywords are forwarded to `AdvancedMH.sample`.

# Examples
```julia
using DistributionsInference, AdvancedMH, Bijectors

chain = distribution_to_chain(
    delay, data, MetropolisEngine(; nsamples = 2000, burnin = 1000))
```

# See also
- [`distribution_to_chain`](@ref): the contract this satisfies.
- [`TuringEngine`](@ref): the sibling engine.
"
struct MetropolisEngine{S, K <: NamedTuple}
    sampler::S
    nsamples::Int
    burnin::Int
    nchains::Int
    proposal_scale::Float64
    kwargs::K
end

function MetropolisEngine(sampler = nothing; nsamples::Integer,
        burnin::Integer = 0, nchains::Integer = 1,
        proposal_scale::Real = 0.1, kwargs...)
    nsamples > 0 || throw(ArgumentError(
        "MetropolisEngine: nsamples must be positive, got $nsamples"))
    0 <= burnin < nsamples || throw(ArgumentError(
        "MetropolisEngine: burnin must satisfy 0 <= burnin < nsamples " *
        "(got burnin=$burnin, nsamples=$nsamples)"))
    nchains > 0 || throw(ArgumentError(
        "MetropolisEngine: nchains must be positive, got $nchains"))
    proposal_scale > 0 || throw(ArgumentError(
        "MetropolisEngine: proposal_scale must be positive, got " *
        "$proposal_scale"))
    return MetropolisEngine(sampler, Int(nsamples), Int(burnin), Int(nchains),
        Float64(proposal_scale), NamedTuple(kwargs))
end

function distribution_to_chain(
        obj, data, engine::MetropolisEngine, rest...; kwargs...)
    throw(ArgumentError(
        "`distribution_to_chain` has no method for a `MetropolisEngine`: " *
        "its implementation lives in the `DistributionsInferenceAdvancedMHExt` " *
        "package extension, which loads only once both `AdvancedMH` and " *
        "`Bijectors` are in the session. Run `using AdvancedMH, Bijectors` " *
        "first (and `Pkg.add` either one that is not installed yet)."))
end
