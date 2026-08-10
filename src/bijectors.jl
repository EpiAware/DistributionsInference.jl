# The unconstrained <-> constrained transform for the PPL-neutral engine, both
# directions and the two things built on them: stubs declared here with their
# docstrings, whose methods live in `DistributionsInferenceBijectorsExt`.
# `distribution_to_objective` is the one function here with no method of its
# own: it is written directly in core over `distribution_to_logdensity` and
# `logdensity_to_objective`, so it needs `Bijectors` only because the latter
# does.

@doc "

Map an unconstrained vector to the constrained scale and its log-Jacobian.

`to_constrained(prob, z)` returns `(x, logjac)`: the constrained estimated flat
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
`ArgumentError`, mirroring [`distribution_to_turing`](@ref)'s rejection of the
same row kind. A type needing an unconstrained transform for such a row
supplies its own [`to_constrained`](@ref) method.

This has no method until `Bijectors` is loaded; the prior-driven transform
lives in the `DistributionsInferenceBijectorsExt` extension.

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
prob = distribution_to_logdensity(leaf, data)
# An unconstrained draw maps to the constrained (positive) shape plus a
# log-Jacobian.
x, logjac = DistributionsInference.to_constrained(prob, [0.0])
x
```

# See also
- [`distribution_to_logdensity`](@ref): assemble `prob`.
- [`logdensity`](@ref): the constrained-scale density this transform feeds.
- [`parameter_rows`](@ref), [`reconstruct`](@ref): the fit protocol this reads.
"
function to_constrained(prob, z)
    return _extension_required(:to_constrained, "Bijectors",
        "DistributionsInferenceBijectorsExt")
end

@doc "

Map constrained estimated parameters to the unconstrained scale.

`to_unconstrained(prob, x)` is the forward direction of
[`to_constrained`](@ref): the unconstrained vector `z` whose constrained image
is `x`, built per row from [`FitLogDensity`](@ref)'s stored `flat_priors` (each
estimated row's `Bijectors.bijector(prior)`). No log-Jacobian comes back, since
the value of the transform is what a caller wants here — most often a starting
point for an optimiser at the parameter values the template already carries,
rather than an arbitrary zero.

An estimated row with no per-row prior (`prior === nothing`, scored instead
through [`extra_logprior`](@ref)) has no distribution to build a bijector from
and is rejected with an `ArgumentError`, exactly as in [`to_constrained`](@ref).

This has no method until `Bijectors` is loaded; the prior-driven transform
lives in the `DistributionsInferenceBijectorsExt` extension.

# Arguments
- `prob`: the assembled [`FitLogDensity`](@ref).
- `x`: a constrained estimated flat vector of length
  [`flat_dimension`](@ref)`(prob.obj)`.

# Examples
```@example
using DistributionsInference, Distributions, Bijectors

struct UnconstrainedLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::UnconstrainedLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

leaf = UnconstrainedLeaf(2.0, 1.0)
prob = distribution_to_logdensity(leaf, [1.5, 2.0, 3.2])
# The positive shape maps onto the whole real line.
DistributionsInference.to_unconstrained(prob, [2.0])
```

# See also
- [`to_constrained`](@ref): the inverse, plus its log-Jacobian.
- [`optimise_distribution`](@ref): the fit that starts from this point.
"
function to_unconstrained(prob, x)
    return _extension_required(:to_unconstrained, "Bijectors",
        "DistributionsInferenceBijectorsExt")
end

@doc "

The negative unconstrained log-posterior, as a plain callable for an
external optimiser.

`logdensity_to_objective(prob)` returns `f(z) -> Real`: the negative of
[`to_constrained`](@ref)'s unconstrained-scale log-target
`logdensity(prob, x) + logjac` at `(x, logjac) = to_constrained(prob, z)`.
Because [`distribution_to_logdensity`](@ref)'s objective is already a plain
(unnormalised) log-density, minimising `f` with any standard optimisation
package finds a maximum a posteriori point directly. `logdensity` always
scores an estimated row's own prior (that is what makes a row estimated; see
[`parameter_rows`](@ref)), so a genuine maximum likelihood point needs a prior
whose curvature is negligible next to the data likelihood rather than a
`loglik` swap alone. The optimiser stays external (`Optim.jl`,
`Optimization.jl`, or any package that accepts a plain callable and an initial
vector); DistributionsInference ships no estimator method itself.

The optimiser's minimiser `z_hat` becomes a fitted object through
[`objective_to_distribution`](@ref)`(prob, z_hat)`. Reach for this function
when the optimiser call is yours to write;
[`optimise_distribution`](@ref) runs the whole round trip instead.

This has no method until `Bijectors` is loaded; the implementation lives in
the `DistributionsInferenceBijectorsExt` extension.

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
prob = distribution_to_logdensity(leaf, data)
f = logdensity_to_objective(prob)
# `f` is a plain callable, ready for any external optimiser.
f([0.0])
```

# See also
- [`to_constrained`](@ref): the unconstrained transform this composes.
- [`distribution_to_logdensity`](@ref), [`logdensity`](@ref): the underlying
  objective.
- [`objective_to_distribution`](@ref): the minimiser back to a fitted object.
"
function logdensity_to_objective(prob)
    return _extension_required(:logdensity_to_objective, "Bijectors",
        "DistributionsInferenceBijectorsExt")
end

@doc "

Build an optimiser objective straight from a distribution and data.

`distribution_to_objective(obj, data; loglik)` is
[`logdensity_to_objective`](@ref)`(`[`distribution_to_logdensity`](@ref)`(obj,
data; loglik))`: the composed convenience so a caller reaches the objective
without assembling a [`FitLogDensity`](@ref) by hand first. It is written in
core over those two functions rather than reimplemented, so it needs
`Bijectors` loaded for the same reason [`logdensity_to_objective`](@ref) does,
and gives the same `ArgumentError` naming `Bijectors` and
`DistributionsInferenceBijectorsExt` when it is not.

Reach for this function when the objective is the whole of what is wanted;
[`optimise_distribution`](@ref) runs the optimiser and rebuilds a fitted
object too, and is a strict superset of what this composes.

# Arguments
- `obj`: the template fittable object, carrying its [`parameter_rows`](@ref).
- `data`: the observed records.

# Keyword Arguments
- `loglik`: a reducer `(obj, data) -> Real` scoring `data` against the
  reconstructed object (default: sum of `logpdf(obj, record)`).

# Examples
```@example
using DistributionsInference, Distributions, Bijectors

struct ComposedLeaf
    shape::Float64
    scale::Float64
end

Distributions.logpdf(d::ComposedLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::ComposedLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ComposedLeaf, x::AbstractVector)
    return ComposedLeaf(x[1], d.scale)
end

leaf = ComposedLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2, 2.8, 1.9]
f = distribution_to_objective(leaf, data)
# `f` is the same callable the two-step route gives, ready for any external
# optimiser.
f([0.0])
```

# See also
- [`logdensity_to_objective`](@ref), [`distribution_to_logdensity`](@ref): the
  two calls this composes.
- [`optimise_distribution`](@ref): the whole fit, objective through to a
  rebuilt object.
"
function distribution_to_objective(obj, data; loglik = _default_loglik)
    prob = distribution_to_logdensity(obj, data; loglik = loglik)
    return logdensity_to_objective(prob)
end

@doc "

Rebuild a fitted distribution from an optimiser's minimiser.

`objective_to_distribution(prob, z)` closes the round trip
[`logdensity_to_objective`](@ref) opens: it maps the unconstrained point `z` an
optimiser returned back to the constrained scale with [`to_constrained`](@ref)
and rebuilds a concrete object there via [`reconstruct`](@ref). What comes back
is the same kind of object `prob` was assembled from, not a parameter vector.

This has no method until `Bijectors` is loaded; the implementation lives in
the `DistributionsInferenceBijectorsExt` extension.

# Arguments
- `prob`: the assembled [`FitLogDensity`](@ref).
- `z`: an unconstrained flat vector of length
  [`flat_dimension`](@ref)`(prob.obj)`, e.g. an optimiser's minimiser.

# Examples
```@example
using DistributionsInference, Distributions, Bijectors

struct MinimiserLeaf
    shape::Float64
    scale::Float64
end

function Distributions.logpdf(d::MinimiserLeaf, y::Real)
    return logpdf(Gamma(d.shape, d.scale), y)
end

function DistributionsInference.parameter_rows(d::MinimiserLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::MinimiserLeaf, x::AbstractVector)
    return MinimiserLeaf(x[1], d.scale)
end

leaf = MinimiserLeaf(2.0, 1.0)
prob = distribution_to_logdensity(leaf, [1.5, 2.0, 3.2])
objective_to_distribution(prob, [0.5]).shape
```

# See also
- [`optimise_distribution`](@ref): the one-call fit this is the last step of.
- [`logdensity_to_objective`](@ref): the objective whose minimiser this reads.
- [`to_constrained`](@ref), [`reconstruct`](@ref): the two steps it composes.
"
function objective_to_distribution(prob, z)
    return _extension_required(:objective_to_distribution, "Bijectors",
        "DistributionsInferenceBijectorsExt")
end
