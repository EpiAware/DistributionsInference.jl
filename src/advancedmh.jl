# Sample a fittable object's posterior with `AdvancedMH`'s random-walk
# Metropolis and return a `FlexiChain`, declared here as a stub with its
# docstring; the method lives in `ext/DistributionsInferenceAdvancedMHExt.jl`
# (triggered by `AdvancedMH` and `Bijectors` together, since it samples on the
# unconstrained scale — see that extension's header comment).

@doc "

Sample a fittable object's posterior with `AdvancedMH`'s random-walk
Metropolis, and return a `FlexiChain`.

`distribution_to_advancedmh(obj, data, sampler, nsamples; nchains = 1,
burnin = 0, loglik, rng, kwargs...)` assembles
[`distribution_to_logdensity`](@ref)`(obj, data; loglik)`, builds an
`AdvancedMH.DensityModel` over its estimated flat parameters on the
*unconstrained* scale (via [`to_constrained`](@ref), so this needs
`Bijectors` loaded as well as `AdvancedMH`), drives it with `sampler` for
`nsamples` draws, and keys the draws back onto the constrained scale into a
dotted-name `FlexiChain` with [`draws_to_chain`](@ref) — the same naming
[`distribution_to_turing`](@ref)'s sampling form produces, so the result
reads back with [`point_estimate`](@ref)/[`inference_to_distribution`](@ref)
unchanged.

Sampling on the unconstrained scale means a random-walk proposal can never
land outside a prior's support in the first place (a negative rate, say),
which a `DensityModel` built directly over the constrained scale would need
a hand-written guard for (`any(<=(0), x) ? -Inf : ...`, the pattern the
getting-started tutorials used before this function existed). This needs
every estimated row to carry its own prior, exactly
[`to_constrained`](@ref)'s own requirement — a row scored instead through
[`extra_logprior`](@ref) (an object-dependent prior, e.g. a centred-pooled
`ComposedDistributions` tree) has no bijector to build, the same row kind
[`distribution_to_turing`](@ref) also refuses. Such an object still fits
through [`distribution_to_logdensity`](@ref) driven by hand, on the
constrained scale, where the hand-written guard is still needed.

`sampler` sizes its own proposal, so a caller who wants the common
`RWMH(MvNormal(zeros(dim), scale^2 * I))` shape reads `dim` off
[`flat_dimension`](@ref DistributionsInference.flat_dimension)`(obj)` first
(see the example below). `burnin` drops that many draws from the start of
each chain before pooling. `nchains > 1` runs that many independent `sample`
calls and pools them into one multi-chain `FlexiChain`, chain-major, the same
convention [`draws_to_chain`](@ref) uses.

This method is available only when both `AdvancedMH` and `Bijectors` are
loaded.

# Arguments
- `obj`: the template fittable object, carrying its [`parameter_rows`](@ref).
- `data`: the observed records scored by `loglik`.
- `sampler`: an `AdvancedMH.MHSampler`, e.g. `RWMH(...)`.
- `nsamples`: the number of samples per chain (including any `burnin`).

# Keyword Arguments
- `nchains`: the number of independent chains to sample (default `1`).
- `burnin`: draws dropped from the start of each chain before pooling
  (default `0`).
- `loglik`: a reducer `(obj, data) -> Real` scoring `data` against the
  reconstructed object (default: sum of `logpdf(obj, record)`), the same
  default [`distribution_to_logdensity`](@ref) uses.
- `rng`: the random-number generator `sample` draws with (default
  `Random.default_rng()`).
- other keywords are forwarded to `AdvancedMH.sample`.

# Examples
```@example
using DistributionsInference, Distributions, AdvancedMH, Bijectors, Random
using LinearAlgebra: I

struct MHGammaLeaf{S <: Real}
    shape::S
    scale::Float64
end

Distributions.logpdf(d::MHGammaLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::MHGammaLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::MHGammaLeaf, x::AbstractVector)
    return MHGammaLeaf(x[1], d.scale)
end

leaf = MHGammaLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2]
dim = DistributionsInference.flat_dimension(leaf)

Random.seed!(1)
chain = distribution_to_advancedmh(
    leaf, data, RWMH(MvNormal(zeros(dim), 0.1^2 * I)), 2000; burnin = 1000)
fitted = point_estimate(leaf, chain)
fitted.scale  # the fixed parameter, untouched
```

# See also
- [`distribution_to_logdensity`](@ref): the PPL-neutral log-density this wraps.
- [`distribution_to_turing`](@ref): the sibling sampling verb.
- [`point_estimate`](@ref) / [`inference_to_distribution`](@ref): read the
  returned chain back onto `obj`.
- [`draws_to_chain`](@ref): the raw-draws-to-chain step this uses internally.
"
function distribution_to_advancedmh end

# The guard for a session missing `AdvancedMH` and/or `Bijectors`. The
# docstring stays on the function binding above; the extension's method takes
# exactly `(obj, data, sampler, nsamples)`, so this trails a `Vararg` to stay
# strictly less specific than it.
function distribution_to_advancedmh(obj, data, rest...; kwargs...)
    throw(ArgumentError(
        "`distribution_to_advancedmh` has no method for these arguments: " *
        "its implementation lives in the `DistributionsInferenceAdvancedMHExt` " *
        "package extension, which loads only once both `AdvancedMH` and " *
        "`Bijectors` are in the session. Run `using AdvancedMH, Bijectors` " *
        "first (and `Pkg.add` either one that is not installed yet)."))
end
