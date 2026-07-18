# The unconstrained <-> constrained transform for the PPL-neutral engine: a
# Turing-free stub declared here (with its docstring), whose method lives in
# the weakdep `DistributionsInferenceBijectorsExt` extension (`ext/`), loaded
# only when `Bijectors` is present. Ported from ComposedDistributions'
# `to_constrained` (ComposedDistributions#185), generalised from a composed
# tree's `flat_priors` to the row-based fit protocol's `FitLogDensity`.

@doc "

Map an unconstrained vector to the constrained scale and its log-Jacobian.

`to_constrained(prob, z)` returns `(x, logjac)`: the constrained ESTIMATED flat
parameters `x` corresponding to the unconstrained vector `z`, and the
log-determinant Jacobian of that (inverse) transform. The transform is built
per row from [`FitLogDensity`](@ref)'s stored `flat_priors` (each estimated
row's `Bijectors.bijector(prior)` — a positive-support prior pushes through an
exp-type link, a simplex-valued prior through a logit/stick-breaking-type
link, and so on). The unconstrained log-density a sampler works with is
`logdensity(prob, x) + logjac`.

An estimated row with no per-row prior (`prior === nothing`, scored instead
through [`extra_logprior`](@ref) — an object-dependent prior, e.g. a
hierarchical population term; see [`parameter_rows`](@ref)) has no
distribution to build a bijector from, so it is rejected with a clear
`ArgumentError`, mirroring [`as_turing`](@ref)'s rejection of the same row
kind. A type needing an unconstrained transform for such a row supplies its
own [`to_constrained`](@ref) method (mirrors ComposedDistributions' centred-pool
handling, which reads the transform off the pooled population's family
instead of the row's own — absent — prior).

This has no method until `Bijectors` is loaded; the prior-driven transform
lives in the `DistributionsInferenceBijectorsExt` extension, so the core
package stays free of a `Bijectors` dependency.

# Arguments
- `prob`: the assembled [`FitLogDensity`](@ref).
- `z`: an unconstrained flat vector of length
  [`flat_dimension`](@ref)`(prob.obj)`.

# Examples
```@example
using DistributionsInference, Distributions, Bijectors

struct ConstrainedLeaf
    shape::Float64
    scale::Float64
end

Distributions.logpdf(d::ConstrainedLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::ConstrainedLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ConstrainedLeaf, x::AbstractVector)
    return ConstrainedLeaf(x[1], d.scale)
end

leaf = ConstrainedLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2]
prob = DistributionsInference.as_logdensity(leaf, data)
# An unconstrained draw maps to the constrained (positive) shape plus a
# log-Jacobian.
x, logjac = DistributionsInference.to_constrained(prob, [0.0])
x
```

# See also
- [`as_logdensity`](@ref): assemble `prob`.
- [`logdensity`](@ref): the constrained-scale density this transform feeds.
- [`parameter_rows`](@ref), [`reconstruct`](@ref): the fit protocol this reads.
"
function to_constrained end
