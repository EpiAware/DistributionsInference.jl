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

# See also
- [`as_logdensity`](@ref): the assembler.
- [`logdensity`](@ref): evaluate on a flat vector.
"
struct FitLogDensity{D, T, L, FP}
    obj::D
    data::T
    loglik::L
    flat_priors::FP
end

function FitLogDensity(obj, data, loglik)
    flat_priors = [row.prior for row in estimated_rows(obj)]
    return FitLogDensity(obj, data, loglik, flat_priors)
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
- [`reconstruct_with_logprior`](@ref): the flat vector -> concrete object plus
  extra-logprior hook this calls (fusing [`reconstruct`](@ref) and
  [`extra_logprior`](@ref) so a package can share work between the two).
"
function logdensity(prob::FitLogDensity, x::AbstractVector)
    flat_priors = prob.flat_priors
    length(x) == length(flat_priors) ||
        _throw_logdensity_dimmismatch(x, flat_priors, prob.obj)
    lp = isempty(x) ? 0.0 :
         sum(_row_logprior(flat_priors[i], x[i]) for i in eachindex(x))
    obj, extra = reconstruct_with_logprior(prob.obj, x)
    lp += extra
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
