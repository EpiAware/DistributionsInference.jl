# A DynamicPPL model over a fittable object's estimated parameters, declared
# here as a stub with its docstring; the model lives in
# `ext/DistributionsInferenceDynamicPPLExt.jl`.

@doc "

A DynamicPPL model over a fittable object's estimated parameters.

`distribution_to_turing(obj, data)` returns a `DynamicPPL`/`Turing` model
whose free parameters are the estimated parameters of `obj` (one `~` site per
[`estimated_rows`](@ref)`(obj)` row, the same flat parameters
[`distribution_to_logdensity`](@ref) exposes), so a fitted posterior is
sampleable with `sample(distribution_to_turing(obj, data), NUTS(), ...)`. It
is a light wrapper on the [`distribution_to_logdensity`](@ref) codec: each
estimated row is a named `~` site sampled from its own `prior`, and the data
likelihood plus [`extra_logprior`](@ref) are added with
`DynamicPPL.@addlogprob!` from the codec's [`reconstruct`](@ref)`(obj, θ)`
scored by `loglik`. The model's total log-density equals
[`logdensity`](@ref)`(distribution_to_logdensity(obj, data), x)` at the
corresponding constrained `x` by construction.

The `~` sites are named to match the
[`point_estimate`](@ref)/[`readback_draws`](@ref) contract exactly: an
estimated row's dotted `name` (e.g. `Symbol(\"onset.shape\")`) becomes the
`VarName` `<prefix>.onset.shape`, so a chain from
`sample(distribution_to_turing(obj, data), ...; chain_type =
FlexiChains.VNChain)` reads back through
[`point_estimate`](@ref)/[`readback_draws`](@ref) unchanged (that
`VarName`-keyed dispatch lives in the
`DistributionsInferenceDynamicPPLFlexiChainsExt` extension, so it needs
`FlexiChains` loaded too; `distribution_to_turing` itself does not).

An estimated row with no fixed `prior` (`prior === nothing`, scored instead
through [`extra_logprior`](@ref) — an object-dependent prior, e.g. a
hierarchical population term; see [`parameter_rows`](@ref)) has no `~` site to
sample it from and is rejected with an `ArgumentError`. Sample such an object
with [`distribution_to_logdensity`](@ref) + `LogDensityProblemsAD` (the
`LogDensityProblems` extension) instead.

A gradient-based sampler (e.g. `NUTS`) evaluates [`reconstruct`](@ref) at a
`ForwardDiff.Dual`-valued flat vector, so each estimated field of `obj`'s type
must be generically typed. A field concretely typed `Float64` errors under
`NUTS`; a gradient-free sampler such as `AdvancedMH` has no such constraint.

This method is available only when `DynamicPPL` is loaded.

# Arguments
- `obj`: the template fittable object, carrying its [`parameter_rows`](@ref).
- `data`: the observed records scored by `loglik`.

# Keyword Arguments
- `prefix`: the outer submodel variable name the sites are namespaced under
  (default `:d`), matching the readback prefix.
- `loglik`: a reducer `(obj, data) -> Real` scoring `data` against the
  reconstructed object (default: sum of `logpdf(obj, record)`), the same
  default [`distribution_to_logdensity`](@ref) uses.

# Examples
```@example
using DistributionsInference, Distributions, DynamicPPL, Turing, Random
using FlexiChains: FlexiChains, VNChain

struct TuringGammaLeaf{S <: Real}
    shape::S
    scale::Float64
end

Distributions.logpdf(d::TuringGammaLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::TuringGammaLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::TuringGammaLeaf, x::AbstractVector)
    return TuringGammaLeaf(x[1], d.scale)
end

leaf = TuringGammaLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2]

Random.seed!(1)
chain = sample(distribution_to_turing(leaf, data), NUTS(), 200;
    chain_type = VNChain, progress = false)
fitted = point_estimate(leaf, chain)
fitted.scale  # the fixed parameter, untouched
```

# See also
- [`distribution_to_logdensity`](@ref): the PPL-neutral log-density this wraps.
- [`point_estimate`](@ref) / [`readback_draws`](@ref): read a fitted chain
  back onto `obj`.
- [`parameter_rows`](@ref) / [`reconstruct`](@ref): the fit protocol this reads.
"
function distribution_to_turing end

# The guard for a session without `DynamicPPL`. The docstring stays on the
# function binding above, so it is declared separately; the extension's method
# takes exactly `(obj, data)`, so this trails a `Vararg` to stay strictly less
# specific than it.
function distribution_to_turing(obj, data, rest...; kwargs...)
    return _extension_required(:distribution_to_turing, "DynamicPPL",
        "DistributionsInferenceDynamicPPLExt")
end

# The `VarName` an estimated row's `~` site carries, given the model `prefix`
# and the row's dotted `name` (e.g. prefix `:d`, name `Symbol("onset.shape")`
# -> `d.onset.shape`). Its method lives in
# `DistributionsInferenceDynamicPPLExt`, but it is declared in the core module
# so the `VarName`-keyed readback in a sibling extension reaches it by
# ordinary dispatch rather than keeping a second copy of the naming contract.
function _row_varname end
