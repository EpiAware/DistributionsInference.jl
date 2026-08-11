# The PPL-neutral log-density engine: assembles a `FitLogDensity` from any
# fit-protocol object (`protocol.jl`) and data, and evaluates its
# (unnormalised) log-posterior over the estimated flat parameter vector.
# Ported from ComposedDistributions (CD#185).

_default_loglik(obj, data) = sum(record -> Distributions.logpdf(obj, record), data)

@doc "

A PPL-neutral log-density over a fit-protocol object's estimated parameters.

`FitLogDensity` carries everything needed to evaluate the (unnormalised)
log-posterior of a fittable object over its estimated flat parameter vector,
with no PPL dependency: the template `obj`, the observed `data`, a `loglik`
reducer scoring `data` against the object reconstructed at a draw, and the
estimated rows' priors flattened once at construction. Build it with
[`distribution_to_logdensity`](@ref); evaluate it on a flat vector with
[`logdensity`](@ref). It also implements the `LogDensityProblems` interface
directly, so it is sampleable by any LogDensityProblems consumer.

# Fields
- `obj`: the template fittable object (the structure [`reconstruct`](@ref)
  rebuilds).
- `data`: the observed records scored by `loglik`.
- `loglik`: a reducer `(obj, data) -> Real` (default sums `logpdf(obj,
  record)`).
- `flat_priors`: the estimated rows' priors, in [`parameter_rows`](@ref) order,
  collected once at construction. An entry is `nothing` for an estimated row
  scored instead through [`extra_logprior`](@ref) (an object-dependent prior;
  see [`parameter_rows`](@ref)), which then contributes no per-row term. A tree
  mixing several prior families makes this vector abstractly typed, costing one
  dynamic dispatch per row in [`logdensity`](@ref).
- `extra_state`: [`extra_prior_state`](@ref)`(obj)`, collected once at
  construction and threaded into every [`extra_logprior`](@ref) call (#28).
- `concrete_fields`: `_concrete_field_candidates(obj)`, the concrete-field-
  under-AD guard's structural state, collected once at construction and empty
  for a properly generic object.

# See also
- [`distribution_to_logdensity`](@ref): the assembler.
- [`logdensity`](@ref): evaluate on a flat vector.
"
struct FitLogDensity{D, T, L, FP, ES, CF}
    obj::D
    data::T
    loglik::L
    flat_priors::FP
    extra_state::ES
    concrete_fields::CF
end

function FitLogDensity(obj, data, loglik)
    rows = estimated_rows(obj)
    flat_priors = [row.prior for row in rows]
    extra_state = extra_prior_state(obj)
    concrete_fields = _concrete_field_candidates(typeof(obj), rows)
    return FitLogDensity(
        obj, data, loglik, flat_priors, extra_state, concrete_fields)
end

@doc "

The template fittable object a [`FitLogDensity`](@ref) was assembled from.

`template(prob)` is the accessor onto `prob`'s `obj` field: the structure
[`reconstruct`](@ref) rebuilds. An engine author reaches it through this
function rather than `prob.obj` directly, so a struct-field rename in a
future release does not break an engine implemented outside this package.

# Arguments
- `prob`: the assembled [`FitLogDensity`](@ref).

# Examples
```@example
using DistributionsInference, Distributions

struct AccessorLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::AccessorLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

leaf = AccessorLeaf(2.0, 1.0)
prob = distribution_to_logdensity(leaf, [1.5, 2.0, 3.2])
DistributionsInference.template(prob) === leaf
```

# See also
- [`observations`](@ref), [`flat_priors`](@ref): the other two accessors.
- [`distribution_to_logdensity`](@ref): assembles `prob`.
"
template(prob::FitLogDensity) = prob.obj

@doc "

The observed data a [`FitLogDensity`](@ref) scores against.

`observations(prob)` is the accessor onto `prob`'s `data` field: the records
`prob`'s `loglik` reducer is scored against at each [`logdensity`](@ref) call.
An engine author reaches it through this function rather than `prob.data`
directly, for the same reason as [`template`](@ref).

# Arguments
- `prob`: the assembled [`FitLogDensity`](@ref).

# Examples
```@example
using DistributionsInference, Distributions

struct AccessorLeaf2
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::AccessorLeaf2)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

leaf = AccessorLeaf2(2.0, 1.0)
data = [1.5, 2.0, 3.2]
prob = distribution_to_logdensity(leaf, data)
DistributionsInference.observations(prob) === data
```

# See also
- [`template`](@ref), [`flat_priors`](@ref): the other two accessors.
- [`distribution_to_logdensity`](@ref): assembles `prob`.
"
observations(prob::FitLogDensity) = prob.data

@doc "

The estimated rows' priors a [`FitLogDensity`](@ref) was assembled with.

`flat_priors(prob)` is the accessor onto `prob`'s `flat_priors` field: the
[`estimated_rows`](@ref)`(template(prob))` priors, collected once at
construction, in [`parameter_rows`](@ref) order. An entry is `nothing` for a
row scored instead through [`extra_logprior`](@ref) (an object-dependent
prior). An engine author reaches it through this function rather than
`prob.flat_priors` directly, for the same reason as [`template`](@ref) — this
is also the vector [`to_constrained`](@ref)/[`to_unconstrained`](@ref) build
their per-row transform from.

# Arguments
- `prob`: the assembled [`FitLogDensity`](@ref).

# Examples
```@example
using DistributionsInference, Distributions

struct AccessorLeaf3
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::AccessorLeaf3)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

leaf = AccessorLeaf3(2.0, 1.0)
prob = distribution_to_logdensity(leaf, [1.5, 2.0, 3.2])
DistributionsInference.flat_priors(prob)
```

# See also
- [`template`](@ref), [`observations`](@ref): the other two accessors.
- [`to_constrained`](@ref), [`to_unconstrained`](@ref): built from this vector.
"
flat_priors(prob::FitLogDensity) = prob.flat_priors

@doc "

Assemble a [`FitLogDensity`](@ref) from a fittable object and data.

`distribution_to_logdensity(obj, data; loglik)` packages the template `obj`
and the observed `data` into the PPL-neutral log-density spec, reading the
priors off `obj`'s [`parameter_rows`](@ref) (the estimation boundary). The
result evaluates the (unnormalised) log-posterior over the estimated flat
parameter vector via [`logdensity`](@ref), on the constrained scale: each
prior is scored directly against its row's value with no Jacobian correction.
An object with no estimated rows estimates nothing: the flat vector is empty
and `logdensity` is just the data likelihood. Sampling on the unconstrained
scale (the transform and its log-Jacobian) is a `Bijectors` extension concern.

A conditionally available exact likelihood is a `loglik` the caller writes and
passes in, not a helper this package adds (#44). Choose between the exact and
approximate branch with an explicit predicate or by dispatch, never by
catching an exception from the exact path, which would hide a genuine bug in
the exact branch. Where the exact form does not apply, refuse loudly with a
named structural reason, the convention [`to_constrained`](@ref) and
[`distribution_to_turing`](@ref) follow for a row kind they do not support.

# Arguments
- `obj`: the template fittable object, carrying its [`parameter_rows`](@ref).
- `data`: the observed records.

# Keyword Arguments
- `loglik`: a reducer `(obj, data) -> Real` scoring `data` against the
  reconstructed object (default: sum of `logpdf(obj, record)`).

# Examples
```@example
using DistributionsInference, Distributions

struct ToyLeaf
    shape::Float64
    scale::Float64
end

Distributions.logpdf(d::ToyLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::ToyLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ToyLeaf, x::AbstractVector)
    return ToyLeaf(x[1], d.scale)
end

leaf = ToyLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2]
prob = distribution_to_logdensity(leaf, data)
DistributionsInference.flat_dimension(leaf)
```

A conditionally exact likelihood, chosen by predicate and passed straight in
as `loglik`:

```@example
using DistributionsInference, Distributions

struct ToyLeaf2
    shape::Float64
    scale::Float64
end

Distributions.logpdf(d::ToyLeaf2, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::ToyLeaf2)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ToyLeaf2, x::AbstractVector)
    return ToyLeaf2(x[1], d.scale)
end

has_exact_form(::ToyLeaf2) = false   # a structural property of the object

function chosen_loglik(obj, data)
    if has_exact_form(obj)
        return sum(y -> logpdf(obj, y), data)  # closed form
    else
        error(\"no exact likelihood for ToyLeaf2: <named structural reason>\")
    end
end

# Decide by an explicit predicate and refuse loudly when it does not apply,
# never by catching an exception from the exact path.
leaf2 = ToyLeaf2(2.0, 1.0)
data2 = [1.5, 2.0, 3.2]
prob2 = distribution_to_logdensity(leaf2, data2; loglik = chosen_loglik)
DistributionsInference.flat_dimension(leaf2)
```

# See also
- [`logdensity`](@ref): evaluate the assembled spec on a flat vector.
- [`parameter_rows`](@ref), [`reconstruct`](@ref): the fit protocol this reads.
"
function distribution_to_logdensity(obj, data; loglik = _default_loglik)
    return FitLogDensity(obj, data, loglik)
end

# `@noinline` keeps the error-message construction (which `show`s `obj`) out
# of the hot evaluation path.
@noinline function _throw_logdensity_dimmismatch(x, flat_priors, obj)
    throw(DimensionMismatch(
        "flat parameter vector has length $(length(x)) but $obj has " *
        "$(length(flat_priors)) estimated parameters"))
end

@doc "

Evaluate a [`FitLogDensity`](@ref) on its estimated flat parameter vector.

`logdensity(prob, x)` is the (unnormalised) log-posterior at the estimated
flat vector `x` (in [`parameter_rows`](@ref)`(prob.obj)` row order restricted
to the estimated rows), on the constrained scale: each prior in `x` is scored
directly, with no Jacobian correction (an unconstrained-scale transform is a
`Bijectors` extension concern). The value is the sum of the priors'
log-densities at `x`, plus [`extra_logprior`](@ref) (an object-dependent
prior term; `0.0` unless `prob.obj` overrides it), plus the data
log-likelihood of the object reconstructed there via [`reconstruct`](@ref).
`x` is [`flat_dimension`](@ref)`(prob.obj)` long — empty when `prob.obj`
estimates nothing, where `logdensity` is just the data likelihood.

# Arguments
- `prob`: the assembled [`FitLogDensity`](@ref).
- `x`: an estimated flat parameter vector of length
  [`flat_dimension`](@ref)`(prob.obj)`.

# Examples
```@example
using DistributionsInference, Distributions

struct FitLeaf
    shape::Float64
    scale::Float64
end

Distributions.logpdf(d::FitLeaf, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::FitLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::FitLeaf, x::AbstractVector)
    return FitLeaf(x[1], d.scale)
end

leaf = FitLeaf(2.0, 1.0)
data = [1.5, 2.0, 3.2]
prob = distribution_to_logdensity(leaf, data)
DistributionsInference.logdensity(prob, [2.5])
```

# See also
- [`distribution_to_logdensity`](@ref): assemble `prob`.
- [`reconstruct`](@ref): the flat vector -> concrete object hook this calls.
"
function logdensity(prob::FitLogDensity, x::AbstractVector)
    flat_priors = prob.flat_priors
    length(x) == length(flat_priors) ||
        _throw_logdensity_dimmismatch(x, flat_priors, prob.obj)
    lp = isempty(x) ? 0.0 :
         sum(_row_logprior(flat_priors[i], x[i]) for i in eachindex(x))
    _check_generic_fields(typeof(prob.obj), prob.concrete_fields, x)
    obj = reconstruct(prob.obj, x)
    lp += extra_logprior(prob.obj, obj, x, prob.extra_state)
    return lp + prob.loglik(obj, prob.data)
end

# A `nothing` prior (a fixed row, or an estimated one scored through
# `extra_logprior`) contributes nothing here.
_row_logprior(prior, xi) = prior === nothing ? zero(xi) : Distributions.logpdf(prior, xi)

# --- LogDensityProblems interface ------------------------------------------

# Gradients are delegated to LogDensityProblemsAD downstream, so only the
# zeroth-order capability is claimed.
function LogDensityProblems.capabilities(::Type{<:FitLogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(prob::FitLogDensity)
    return flat_dimension(prob.obj)
end

# The unqualified `logdensity` on the right is this module's own evaluator,
# not the `LogDensityProblems` function being defined.
function LogDensityProblems.logdensity(prob::FitLogDensity, x::AbstractVector)
    return logdensity(prob, x)
end
