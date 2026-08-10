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
- `init`: the starting point on the CONSTRAINED scale — the units `obj`'s own
  [`parameter_rows`](@ref) values are in, and a caller thinks in (default:
  `obj`'s own estimated values). Mapped through [`to_unconstrained`](@ref)
  internally before the optimiser sees it. A value on the boundary of its
  prior's support (a zero rate, a zero or one probability) has no
  unconstrained image, and is rejected with the offending row named rather
  than passed to the optimiser.
- other keywords are forwarded to [`minimise`](@ref), and from there to the
  optimiser package. With `Optim`, `options` takes an `Optim.Options` and
  `autodiff` an `ADTypes` backend; see [`minimise`](@ref) for why a
  gradient-based fit wants the latter set.

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
    # `init` is on the constrained scale (the units `obj`'s own rows are in);
    # mapped through `to_unconstrained` here so the optimiser always sees an
    # unconstrained start, whether it is the template's own values or a
    # caller-given one. A value strictly outside its prior's support (e.g.
    # `-Inf` for a positive-support row) sends a link function like `log`
    # negative, which raises a `DomainError` rather than returning `-Inf`;
    # caught and folded into the same non-finite-start rejection below, so
    # every out-of-support start gets the one named error regardless of
    # which failure mode the link function happens to take.
    x0 = init === nothing ? _template_values(prob.obj) : init
    z0 = try
        to_unconstrained(prob, x0)
    catch e
        e isa DomainError ? fill(-Inf, length(x0)) : rethrow()
    end
    all(isfinite, z0) || _reject_nonfinite_start(prob, z0, init === nothing)
    z_hat = minimise(logdensity_to_objective(prob), z0, optimiser; kwargs...)
    return objective_to_distribution(prob, z_hat)
end

# A constrained value sitting on the boundary of its prior's support (a zero
# rate, a zero or one probability) has no unconstrained image, so the mapped
# start is `+/-Inf` and the optimiser fails on a non-finite objective. Name
# the row instead, since the fault is in the starting point rather than the
# search.
function _reject_nonfinite_start(prob, z0, defaulted)
    rows = estimated_rows(prob.obj)
    bad = findall(!isfinite, z0)
    named = length(rows) == length(z0) ?
            join(("$(rows[i].name) = $(rows[i].value)" for i in bad), ", ") :
            "flat coordinates $(bad)"
    source = defaulted ?
             "the template's own values map to a non-finite point at" :
             "the `init` given maps to a non-finite point at"
    throw(ArgumentError(
        "`optimise_distribution` cannot start the optimiser: $source " *
        "$named. A constrained value on the boundary of its prior's " *
        "support has no unconstrained image; move it inside the support, " *
        "or pass a finite constrained `init`."))
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
is loaded. Its `options` keyword takes an `Optim.Options`, and its `autodiff`
keyword an `ADTypes` backend.

Set `autodiff` for any gradient-based optimiser. `Optim` differentiates by
central finite differences unless told otherwise, which spends `2n` objective
evaluations on every gradient and loses accuracy;
[`logdensity_to_objective`](@ref)'s objective differentiates directly, so an AD
backend gets the same point for a fraction of the work. The backend's package
has to be loaded for `ADTypes` to dispatch on it.

```julia
using DistributionsInference, Bijectors, Optim, ADTypes, ForwardDiff

optimise_distribution(leaf, data, LBFGS(); autodiff = AutoForwardDiff())
```

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
    plugs_in = "Another optimisation package plugs in by adding a " *
               "`minimise` method for its own optimiser type."
    # With the extension already loaded, the fault is the argument rather
    # than a missing package, so telling the caller to install `Optim` would
    # send them after something they have.
    _optim_loaded() && throw(ArgumentError(
        "`minimise` has no method for an optimiser of type " *
        "$(typeof(optimiser)): it is not a supported optimiser. `Optim` is " *
        "loaded, so every `Optim.AbstractOptimizer` (e.g. `LBFGS()`) " *
        "already has one. $plugs_in"))
    throw(ArgumentError(
        "`minimise` has no method for an optimiser of type " *
        "$(typeof(optimiser)): the `Optim.jl` methods live in the " *
        "`DistributionsInferenceOptimExt` package extension, which loads " *
        "only once `Optim` is in the session (run `using Optim`, and " *
        "`Pkg.add(\"Optim\")` if it is not installed yet). $plugs_in"))
end

function _optim_loaded()
    Base.get_extension(
        DistributionsInference, :DistributionsInferenceOptimExt) !== nothing
end
