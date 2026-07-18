# A DynamicPPL model over a fittable object's ESTIMATED parameters, built as
# a light wrapper on the `as_logdensity` codec. Declared here with no method
# (a Turing-free stub); the model lives in
# `ext/DistributionsInferenceDynamicPPLExt.jl`, triggered by `DynamicPPL`
# alone, so this package stays Turing-free until that extension loads. Ported
# from ComposedDistributions' `as_turing` (ComposedDistributions#185),
# generalised from a composed tree's nested edge path to the row-based fit
# protocol.

@doc "

A DynamicPPL model over a fittable object's estimated parameters.

`as_turing(obj, data)` returns a `DynamicPPL`/`Turing` model whose free
parameters are the ESTIMATED parameters of `obj` (one `~` site per
[`estimated_rows`](@ref)`(obj)` row, the same flat parameters
[`as_logdensity`](@ref) exposes), so a fitted posterior is sampleable with
`sample(as_turing(obj, data), NUTS(), ...)`. It is a light wrapper on the
[`as_logdensity`](@ref) codec: each estimated row is a named `~` site sampled
from its own `prior`, and the data likelihood plus [`extra_logprior`](@ref)
are added with `DynamicPPL.@addlogprob!` from the codec's
[`reconstruct`](@ref)`(obj, θ)` scored by `loglik`. The model's total
log-density equals [`logdensity`](@ref)`(as_logdensity(obj, data), x)` at the
corresponding constrained `x` by construction.

The `~` sites are named to match the [`readback`](@ref)/[`readback_draws`](@ref)
contract exactly: an estimated row's dotted `name` (e.g.
`Symbol(\"onset.shape\")`) becomes the `VarName` `<prefix>.onset.shape`, so a
chain from `sample(as_turing(obj, data), ...; chain_type =
FlexiChains.VNChain)` reads back through [`readback`](@ref)/[`readback_draws`](@ref)
unchanged (the `VarName`-keyed dispatch also lives in this extension).

An estimated row with no fixed `prior` (`prior === nothing`, scored instead
through [`extra_logprior`](@ref) — an object-dependent prior, e.g. a
hierarchical population term; see [`parameter_rows`](@ref)) has no `~` site to
sample it from and is rejected with an `ArgumentError`. Sample such an object
with [`as_logdensity`](@ref) + `LogDensityProblemsAD` (the `LogDensityProblems`
extension) instead.

A gradient-based sampler (e.g. `NUTS`) evaluates [`reconstruct`](@ref) at a
`ForwardDiff.Dual`-valued flat vector, so `obj`'s type must accept a non-`Real`
concrete element for each ESTIMATED field: a `Distributions.jl` leaf already
does (its type parameter is generic), and a hand-written `struct` needs the
same — a field concretely typed `Float64` errors under `NUTS` (a
gradient-free sampler, e.g. `AdvancedMH`, has no such constraint).

This method is available only when `DynamicPPL` is loaded (the model lives in a
package extension).

# Arguments
- `obj`: the template fittable object, carrying its [`parameter_rows`](@ref).
- `data`: the observed records scored by `loglik`.

# Keyword Arguments
- `prefix`: the outer submodel variable name the sites are namespaced under
  (default `:d`), matching the readback prefix.
- `loglik`: a reducer `(obj, data) -> Real` scoring `data` against the
  reconstructed object (default: sum of `logpdf(obj, record)`), the same
  default [`as_logdensity`](@ref) uses.

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
chain = sample(as_turing(leaf, data), NUTS(), 200;
    chain_type = VNChain, progress = false)
fitted = DistributionsInference.readback(leaf, chain)
fitted.scale  # the fixed parameter, untouched
```

# See also
- [`as_logdensity`](@ref): the PPL-neutral log-density this wraps.
- [`readback`](@ref) / [`readback_draws`](@ref): read a fitted chain back onto `obj`.
- [`parameter_rows`](@ref) / [`reconstruct`](@ref): the fit protocol this reads.
"
function as_turing end
