# The one-call optimiser fit. `optimise_distribution` composes the pieces a
# point fit needs — the unconstrained starting point, the objective, the
# optimiser call, the map back to a distribution — so a caller writes none of
# them. Each piece stays reachable on its own.
#
# Only the optimiser call itself is package-specific, and that is `minimise`,
# the one hook an optimiser package's extension fills in.

# The template's own estimated values, in estimated-row order: where a fit
# starts unless the caller names another point.
_template_values(obj) = [row.value for row in estimated_rows(obj)]

@doc "

Fit a distribution to data with an optimiser, in one call.

`optimise_distribution(obj, data, optimiser)` runs the whole point fit and
returns a fitted distribution of the same kind as `obj`: it assembles
[`distribution_to_logdensity`](@ref)`(obj, data)`, starts the optimiser at
`obj`'s own parameter values mapped through [`to_unconstrained`](@ref),
minimises [`logdensity_to_objective`](@ref) with `optimiser` via
[`minimise`](@ref), and maps the minimiser back with
[`objective_to_distribution`](@ref). A [`FitLogDensity`](@ref) already to hand
is fitted directly with `optimise_distribution(prob, optimiser)`.

The point found is a maximum a posteriori one on the unconstrained scale, since
[`logdensity`](@ref) always scores an estimated row's own prior (that is what
makes a row estimated; see [`parameter_rows`](@ref)) and the objective carries
the transform's log-Jacobian. A maximum likelihood point needs a prior whose
curvature is negligible next to the data likelihood. An object estimating
nothing has nothing to optimise, and comes back as it went in.

The optimiser package stays a caller's choice: `optimiser` is passed to
[`minimise`](@ref), whose `Optim.jl` method lives in the
`DistributionsInferenceOptimExt` extension. The transform needs `Bijectors`
loaded as well.

# Arguments
- `obj`: the template fittable object, carrying its [`parameter_rows`](@ref).
- `data`: the observed records.
- `optimiser`: the optimiser to minimise with, e.g. `Optim.LBFGS()`.

# Keyword Arguments
- `loglik`: a reducer `(obj, data) -> Real` scoring `data` against the
  reconstructed object (default: sum of `logpdf(obj, record)`).
- `init`: the starting point on the unconstrained scale (default: `obj`'s own
  estimated values through [`to_unconstrained`](@ref)).
- other keywords are forwarded to [`minimise`](@ref), and from there to the
  optimiser package.

# Examples
```@example
using DistributionsInference, Distributions, Bijectors, Optim

struct FitLeaf{T <: Real}
    shape::T
    scale::Float64
end

Distributions.logpdf(d::FitLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::FitLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.5), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::FitLeaf, x::AbstractVector)
    return FitLeaf(x[1], d.scale)
end

data = [1.5, 2.0, 3.2, 2.8, 1.9, 4.1, 2.5, 3.0]
fitted = optimise_distribution(FitLeaf(2.0, 1.0), data, LBFGS())
fitted.shape
```

# See also
- [`minimise`](@ref): the optimiser hook this calls.
- [`logdensity_to_objective`](@ref), [`to_unconstrained`](@ref),
  [`objective_to_distribution`](@ref): the pieces this composes.
- [`distribution_to_logdensity`](@ref): the sampling route to the same
  posterior.
"
function optimise_distribution(obj, data, optimiser; loglik = _default_loglik,
        init = nothing, kwargs...)
    prob = distribution_to_logdensity(obj, data; loglik = loglik)
    return optimise_distribution(prob, optimiser; init = init, kwargs...)
end

function optimise_distribution(
        prob::FitLogDensity, optimiser; init = nothing, kwargs...)
    # Nothing is estimated, so the template already is the fit and no
    # transform (and no `Bijectors`) is needed to say so.
    flat_dimension(prob.obj) == 0 && return reconstruct(prob.obj, Float64[])
    z0 = init === nothing ?
         to_unconstrained(prob, _template_values(prob.obj)) : init
    z_hat = minimise(logdensity_to_objective(prob), z0, optimiser; kwargs...)
    return objective_to_distribution(prob, z_hat)
end

@doc "

Minimise a plain callable from a starting point, with a given optimiser.

`minimise(objective, init, optimiser)` returns the minimising vector. It is the
one optimiser-package-specific step of [`optimise_distribution`](@ref), kept
apart so that supporting another optimisation package is a method on this
function and nothing else. Keyword arguments are forwarded to the optimiser
package's own entry point.

The `Optim.jl` method (any `Optim.AbstractOptimizer`, e.g. `LBFGS()`) lives in
the `DistributionsInferenceOptimExt` extension and has no method until `Optim`
is loaded. Its `options` keyword takes an `Optim.Options`.

# Arguments
- `objective`: a callable `z -> Real` to minimise, e.g. the result of
  [`logdensity_to_objective`](@ref).
- `init`: the starting vector.
- `optimiser`: the optimiser, whose type selects the method.

# Examples
```@example
using DistributionsInference, Optim

DistributionsInference.minimise(z -> sum(abs2, z .- 2.0), [0.0, 0.0], LBFGS())
```

# See also
- [`optimise_distribution`](@ref): the one-call fit this serves.
- [`logdensity_to_objective`](@ref): the objective it usually minimises.
"
function minimise(objective, init, optimiser; kwargs...)
    throw(ArgumentError(
        "`minimise` has no method for an optimiser of type " *
        "$(typeof(optimiser)): the `Optim.jl` methods live in the " *
        "`DistributionsInferenceOptimExt` package extension, which loads " *
        "only once `Optim` is in the session (run `using Optim`, and " *
        "`Pkg.add(\"Optim\")` if it is not installed yet). Another " *
        "optimisation package plugs in by adding a `minimise` method for " *
        "its own optimiser type."))
end
