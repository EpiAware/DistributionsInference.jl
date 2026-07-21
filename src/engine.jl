# The PPL-neutral log-density engine: assembles a `FitLogDensity` from any
# fit-protocol object (`protocol.jl`) and data, and evaluates its
# (unnormalised) log-posterior over the estimated flat parameter vector.
# `LogDensityProblems` is a hard dependency here, so the interface is
# implemented directly on `FitLogDensity` — no glue extension is needed, unlike
# ComposedDistributions' weakdep `LogDensityProblemsExt`. Ported from
# ComposedDistributions' `ComposedLogDensity`/`as_logdensity`/`logdensity`
# (ComposedDistributions#185), generalised from a composed distribution's
# nested tree to the row-based fit protocol.

# Default likelihood: sum `logpdf(obj, record)` over the observed records.
_default_loglik(obj, data) = sum(record -> Distributions.logpdf(obj, record), data)

@doc "

A PPL-neutral log-density over a fit-protocol object's estimated parameters.

`FitLogDensity` carries everything needed to evaluate the (unnormalised)
log-posterior of a fittable object over its ESTIMATED flat parameter vector,
with no PPL dependency: the template `obj`, the observed `data`, a `loglik`
reducer scoring `data` against the object reconstructed at a draw, and the
estimated rows' priors flattened once at construction. Build it with
[`as_logdensity`](@ref); evaluate it on a flat vector with
[`logdensity`](@ref). It also implements the `LogDensityProblems` interface
directly, so it is sampleable by any LogDensityProblems consumer.

# Fields
- `obj`: the template fittable object (the structure [`reconstruct`](@ref)
  rebuilds).
- `data`: the observed records scored by `loglik`.
- `loglik`: a reducer `(obj, data) -> Real` (default sums `logpdf(obj,
  record)`).
- `flat_priors`: the estimated rows' priors, in [`parameter_rows`](@ref) order,
  collected once at construction so [`logdensity`](@ref) does not re-derive
  them on every evaluation. An entry is `nothing` for an estimated row scored
  instead through [`extra_logprior`](@ref) (an object-dependent prior; see
  [`parameter_rows`](@ref)), which then contributes no per-row term. A tree
  mixing several prior families (a `LogNormal` here, a `Beta` there) makes
  this vector abstractly typed, so [`logdensity`](@ref) pays one dynamic
  dispatch per row; a tuple-of-priors specialisation is the natural
  follow-up if profiling ever shows this matters.
- `extra_state`: [`extra_prior_state`](@ref)`(obj)`, collected once at
  construction alongside `flat_priors` and threaded into every
  [`extra_logprior`](@ref) call, so a package whose extra term needs a
  structural walk of `obj` (DI#28's motivating case: which rows carry an
  object-dependent prior) pays it once here rather than on every evaluation.
- `concrete_fields`: `_concrete_field_candidates(obj)`, collected once at
  construction alongside `flat_priors`/`extra_state` — the concrete-field-
  under-AD guard's (DI#48) own structural state, empty for a properly
  generic object, so [`logdensity`](@ref)'s per-evaluation guard is a single
  `isempty` check rather than a fresh `estimated_rows(obj)` walk on every
  call.

# See also
- [`as_logdensity`](@ref): the assembler.
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

Assemble a [`FitLogDensity`](@ref) from a fittable object and data.

`as_logdensity(obj, data; loglik)` packages the template `obj` and the
observed `data` into the PPL-neutral log-density spec, reading the priors off
`obj`'s [`parameter_rows`](@ref) (the estimation boundary). The result
evaluates the (unnormalised) log-posterior over the ESTIMATED flat parameter
vector via [`logdensity`](@ref), on the CONSTRAINED scale: each prior is
scored directly against its row's value with no Jacobian correction. An
object with no estimated rows estimates nothing: the flat vector is empty and
`logdensity` is just the data likelihood. Sampling on the unconstrained scale
(the transform and its log-Jacobian) is a `Bijectors` extension concern, not
this core engine's, mirroring ComposedDistributions' `to_constrained`.

A CONDITIONALLY available exact likelihood — some objects can score their
data through a closed form only under a structural condition, and otherwise
only through an approximation — is a `loglik` a caller writes and passes in
directly, not a helper this package adds (DistributionsInference#44: the
hook plus this convention are enough on their own). Choose between the exact
and approximate branch with an explicit predicate or by dispatch; NEVER by
catching an exception thrown from the exact path, since that hides a genuine
bug in the exact branch as if the exact form simply did not apply. Where the
exact form does not apply, refuse loudly with a named structural reason (an
`error` or `ArgumentError` naming what is missing), the same convention
[`to_constrained`](@ref) and [`as_turing`](@ref) follow for a row kind they
do not support.

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
prob = DistributionsInference.as_logdensity(leaf, data)
DistributionsInference.flat_dimension(leaf)
```

A conditionally exact likelihood, chosen by predicate and passed straight in
as `loglik` (no dedicated helper needed):

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

# GOOD: decide by an explicit predicate, refuse loudly when it does not
# apply. NEVER decide by catching an exception from the exact path (a
# genuine bug in the exact branch would then be silently misread as \"exact
# form unavailable\").
leaf2 = ToyLeaf2(2.0, 1.0)
data2 = [1.5, 2.0, 3.2]
prob2 = DistributionsInference.as_logdensity(
    leaf2, data2; loglik = chosen_loglik)
DistributionsInference.flat_dimension(leaf2)
```

# See also
- [`logdensity`](@ref): evaluate the assembled spec on a flat vector.
- [`parameter_rows`](@ref), [`reconstruct`](@ref): the fit protocol this reads.
"
function as_logdensity(obj, data; loglik = _default_loglik)
    return FitLogDensity(obj, data, loglik)
end

# Hoisted into its own `@noinline` function so the error-message construction
# (which interpolates `obj` via `show`) stays out of the hot evaluation path;
# mirrors ComposedDistributions' `_throw_logdensity_dimmismatch`.
@noinline function _throw_logdensity_dimmismatch(x, flat_priors, obj)
    throw(DimensionMismatch(
        "flat parameter vector has length $(length(x)) but $obj has " *
        "$(length(flat_priors)) estimated parameters"))
end

@doc "

Evaluate a [`FitLogDensity`](@ref) on its estimated flat parameter vector.

`logdensity(prob, x)` is the (unnormalised) log-posterior at the estimated
flat vector `x` (in [`parameter_rows`](@ref)`(prob.obj)` row order restricted
to the estimated rows), on the CONSTRAINED scale: each prior in `x` is scored
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
prob = DistributionsInference.as_logdensity(leaf, data)
DistributionsInference.logdensity(prob, [2.5])
```

# See also
- [`as_logdensity`](@ref): assemble `prob`.
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

# A single row's per-row prior contribution: `nothing` (a fixed parameter, or
# an ESTIMATED one scored instead through `extra_logprior`, per the
# `parameter_rows` convention) contributes nothing here; any other prior
# scores directly against its flat value.
_row_logprior(prior, xi) = prior === nothing ? zero(xi) : Distributions.logpdf(prior, xi)

# --- LogDensityProblems interface (hard dep; no glue extension needed) -----

# The engine supplies the log-density itself; a gradient is delegated to
# LogDensityProblemsAD downstream, so only the zeroth-order capability is
# claimed here.
function LogDensityProblems.capabilities(::Type{<:FitLogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(prob::FitLogDensity)
    return flat_dimension(prob.obj)
end

# Qualified call on the right-hand side: `logdensity` above is this module's
# own evaluator, distinct from `LogDensityProblems.logdensity` being defined.
function LogDensityProblems.logdensity(prob::FitLogDensity, x::AbstractVector)
    return logdensity(prob, x)
end
