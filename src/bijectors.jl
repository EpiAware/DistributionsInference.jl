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

@doc "

The NEGATIVE unconstrained log-posterior, as a plain callable for an
external optimiser.

`as_optimisation_objective(prob)` returns `f(z) -> Real`: the negative of
[`to_constrained`](@ref)'s unconstrained-scale log-target
`logdensity(prob, x) + logjac` at `(x, logjac) = to_constrained(prob, z)`.
Because [`as_logdensity`](@ref)'s objective is already a plain (unnormalised)
log-density, minimising `f` with any standard optimisation package finds a
maximum-A-POSTERIORI point directly. `logdensity` always scores an ESTIMATED
row's own prior (that is what makes a row estimated; see
[`parameter_rows`](@ref)), so a genuine maximum-LIKELIHOOD point needs a
prior whose curvature is negligible next to the data likelihood (a very
diffuse prior on an otherwise ordinary row) rather than a `loglik` swap
alone. DistributionsInference ships no estimator method itself: this is only
the thin wiring of the transform, the objective and [`reconstruct`](@ref)
together that the org's fitting layer needs to earn its place alongside a
bespoke `distribution_mle`/`distribution_map` pair — the optimiser stays
external (`Optim.jl`, `Optimization.jl`, or any package that accepts a plain
callable and an initial vector).

The result reconstructs through the EXISTING readback path: run the
optimiser's minimiser `z_hat` back through `to_constrained(prob, z_hat)` to
recover the constrained point, then [`reconstruct`](@ref)`(prob.obj, x_hat)`
for the fitted object, exactly as after any other unconstrained draw.

This has no method until `Bijectors` is loaded (built directly on
[`to_constrained`](@ref)); the implementation lives in the
`DistributionsInferenceBijectorsExt` extension, so the core package stays
free of a `Bijectors` dependency, and no optimisation package is a dependency
at all.

# Arguments
- `prob`: the assembled [`FitLogDensity`](@ref).

# Examples
```@example
using DistributionsInference, Distributions, Bijectors

struct OptimLeaf
    shape::Float64
    scale::Float64
end

Distributions.logpdf(d::OptimLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::OptimLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::OptimLeaf, x::AbstractVector)
    return OptimLeaf(x[1], d.scale)
end

leaf = OptimLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2, 2.8, 1.9]
prob = DistributionsInference.as_logdensity(leaf, data)
f = DistributionsInference.as_optimisation_objective(prob)
# NOTHING below is a DistributionsInference method: `f` is just a plain
# callable, ready for any external optimiser.
f([0.0])
```

# See also
- [`to_constrained`](@ref): the unconstrained transform this composes.
- [`as_logdensity`](@ref), [`logdensity`](@ref): the underlying objective.
- [`reconstruct`](@ref): rebuild the fitted object from the optimiser's
  minimiser, via [`to_constrained`](@ref) back to the constrained scale.
"
function as_optimisation_objective end
