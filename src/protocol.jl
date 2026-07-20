# The fit protocol: the small public surface a fittable object implements so
# the log-density engine (`engine.jl`) can assemble and evaluate a posterior
# over its estimated parameters. A package implements it by adding methods to
# `parameter_rows` and `reconstruct` on its own type — DistributionsInference
# need not be loaded at that end (plain method extension on our public
# functions, ComposedDistributions#185).
#
# The shape generalises ComposedDistributions' `params_table`: a flat
# inventory of an object's scalar parameters, each row carrying a name, its
# current value, an optional attached prior, and its support. A row whose
# `prior` is not `nothing` is ESTIMATED; the flat vector the engine works over
# spans exactly the estimated rows, in `parameter_rows` order.

@doc "

The scalar parameter rows of a fittable object.

`parameter_rows(obj)` returns an iterable of rows, one per scalar parameter of
`obj`, each a `NamedTuple` with fields:

- `name`: a `Symbol` identifying the parameter (a dotted path for a nested
  parameter, e.g. `Symbol(\"onset.shape\")`).
- `value`: the parameter's current value.
- `prior`: the attached prior (a `UnivariateDistribution`) if the parameter is
  ESTIMATED, or `nothing` if it is fixed at `value`.
- `support`: the `(lower, upper)` bounds of the parameter's admissible domain.

A parameter estimated under an OBJECT-DEPENDENT prior (e.g. a hierarchical
population term whose log-density depends on other reconstructed parameters,
not on `value` alone) also carries `prior = nothing` at the row level — the
same as a fixed parameter — and is scored instead through
[`extra_logprior`](@ref). This keeps the row schema to exactly these four
fields for every parameter kind. A type with such rows must give its own
[`estimated_rows`](@ref)/[`flat_dimension`](@ref) methods (the generic
defaults below treat `prior === nothing` as fixed and would otherwise drop it
from the flat vector).

This is the one function every fittable object implements with its own
method; the fallback below only raises a clear error naming the missing
method. [`estimated_rows`](@ref), [`flat_dimension`](@ref) and the engine's
[`as_logdensity`](@ref) are all built on it. A bare `AbstractVector` of
already-built rows is its own `parameter_rows` (the identity), so a literal
row list can stand in for a fittable object without a wrapping type.

# Arguments
- `obj`: the fittable object.

# Examples
```@example
using DistributionsInference, Distributions

rows = [(name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
        support = (0.0, Inf)),
    (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]
DistributionsInference.parameter_rows(rows) === rows
```

# See also
- [`estimated_rows`](@ref): the subset with an attached prior.
- [`reconstruct`](@ref): the companion hook, flat vector -> concrete object.
"
function parameter_rows(obj)
    throw(ArgumentError(
        "no `parameter_rows` method for $(typeof(obj)); implement the fit " *
        "protocol's `parameter_rows(obj)` for this type"))
end

parameter_rows(rows::AbstractVector{<:NamedTuple}) = rows

@doc "

The ESTIMATED rows of a fittable object: those with a non-`nothing` `prior`.

`estimated_rows(obj)` filters [`parameter_rows`](@ref)`(obj)` to the rows whose
`prior` field is set, in the same order. These are the free parameters the
engine estimates; a fixed row (`prior === nothing`) is excluded. An object with
no estimated rows has [`flat_dimension`](@ref) zero.

# Arguments
- `obj`: the fittable object.

# Examples
```@example
using DistributionsInference, Distributions

rows = [(name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
        support = (0.0, Inf)),
    (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]
DistributionsInference.estimated_rows(rows)
```

# See also
- [`parameter_rows`](@ref): the full row inventory this filters.
- [`flat_dimension`](@ref): the estimated count.
"
function estimated_rows(obj)
    return filter(row -> row.prior !== nothing, collect(parameter_rows(obj)))
end

@doc "

The estimated parameter dimension of a fittable object.

`flat_dimension(obj)` is the number of scalar ESTIMATED parameters: the count
of [`parameter_rows`](@ref)`(obj)` rows whose `prior` is not `nothing`. An
object with no estimated rows has flat dimension 0. It is the length of the
flat vector [`reconstruct`](@ref) consumes and the engine's
[`logdensity`](@ref) evaluates on.

# Arguments
- `obj`: the fittable object.

# Examples
```@example
using DistributionsInference, Distributions

rows = [(name = :shape, value = 2.0, prior = LogNormal(0.0, 0.2),
        support = (0.0, Inf)),
    (name = :scale, value = 1.0, prior = nothing, support = (0.0, Inf))]
DistributionsInference.flat_dimension(rows)
```

# See also
- [`estimated_rows`](@ref): the rows this counts.
"
function flat_dimension(obj)
    return count(row -> row.prior !== nothing, parameter_rows(obj))
end

@doc "

Reconstruct a concrete fittable object from an estimated flat parameter
vector.

`reconstruct(obj, x)` returns a new object of the same kind as `obj` with each
ESTIMATED parameter (an [`estimated_rows`](@ref)`(obj)` row, in
[`parameter_rows`](@ref) order) taken from `x`, and every fixed parameter held
at its value in `obj`. `x` is [`flat_dimension`](@ref)`(obj)` long — empty when
`obj` estimates nothing, in which case `reconstruct(obj, x) == obj`.

This is the companion hook every fittable object implements with its own
method, alongside [`parameter_rows`](@ref); rebuilding a concrete object is
necessarily type-specific, so the fallback below only raises a clear error
naming the missing method. The engine's [`logdensity`](@ref) calls it once
per evaluation to score `prob.data` against the object collapsed at `x`.

# Arguments
- `obj`: the fittable object whose structure is rebuilt.
- `x`: an estimated flat vector of length [`flat_dimension`](@ref)`(obj)`.

# Examples
```@example
using DistributionsInference, Distributions

struct DemoLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::DemoLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::DemoLeaf, x::AbstractVector)
    return DemoLeaf(x[1], d.scale)
end

DistributionsInference.reconstruct(DemoLeaf(2.0, 1.0), [3.5])
```

# See also
- [`parameter_rows`](@ref): the row inventory whose order fixes `x`'s layout.
- [`as_logdensity`](@ref): the engine assembler built on `reconstruct`.
"
function reconstruct(obj, x::AbstractVector)
    throw(ArgumentError(
        "no `reconstruct` method for $(typeof(obj)); implement the fit " *
        "protocol's `reconstruct(obj, x::AbstractVector)` for this type"))
end

# A bare row vector is its own minimal fittable object (the `parameter_rows`
# identity above), so it needs its own `reconstruct`: substitute the
# estimated rows' values from `x`, in row order, and hold every fixed row
# unchanged. Built with an explicit indexed loop (mirrors
# ComposedDistributions' `unflatten`), since the rows are typically
# heterogeneously typed (a `prior` of `Nothing` here, a distribution there).
function reconstruct(rows::AbstractVector{<:NamedTuple}, x::AbstractVector)
    n = count(row -> row.prior !== nothing, rows)
    length(x) == n || throw(DimensionMismatch(
        "flat vector has length $(length(x)) but $(length(rows)) row(s) " *
        "carry $n estimated parameter(s)"))
    out = Vector{Any}(undef, length(rows))
    j = 0
    for i in eachindex(rows)
        row = rows[i]
        if row.prior === nothing
            out[i] = row
        else
            j += 1
            out[i] = merge(row, (value = x[j],))
        end
    end
    return out
end

@doc "

Additional log-prior mass that depends on the RECONSTRUCTED object.

`extra_logprior(obj, reconstructed, x)` is the neutral extension point for a
prior term that cannot be scored per-row against `x` alone — a hierarchical
population term is the motivating case, where a pooled member's log-density
depends on the (reconstructed) population hyperparameters, not just on its
own flat coordinate. The default returns `0.0`: most fittable objects need no
such term, since an ordinary per-parameter prior is already scored from
[`parameter_rows`](@ref)`(obj)`'s `prior` column in the engine's
[`logdensity`](@ref). A type with an object-dependent prior overrides this
with its own method and gives the corresponding row(s) `prior = nothing` (see
[`parameter_rows`](@ref)).

# Arguments
- `obj`: the template fittable object.
- `reconstructed`: `obj` rebuilt at `x` (i.e. [`reconstruct`](@ref)`(obj,
  x)`), the object the extra term is scored against.
- `x`: the estimated flat parameter vector `reconstructed` was built from.

# Examples
```@example
using DistributionsInference, Distributions

struct PooledPair
    a::Float64
    b::Float64
    mu::Float64
end

function DistributionsInference.parameter_rows(p::PooledPair)
    return [(name = :mu, value = p.mu, prior = Normal(0.0, 1.0),
            support = (-Inf, Inf)),
        (name = :a, value = p.a, prior = nothing, support = (-Inf, Inf)),
        (name = :b, value = p.b, prior = nothing, support = (-Inf, Inf))]
end

function DistributionsInference.reconstruct(p::PooledPair, x::AbstractVector)
    return PooledPair(p.a, p.b, x[1])
end

# a and b share the population Normal(mu, 1): an object-dependent prior,
# scored here rather than per row.
function DistributionsInference.extra_logprior(p::PooledPair, r, x)
    return logpdf(Normal(r.mu, 1.0), r.a) + logpdf(Normal(r.mu, 1.0), r.b)
end

DistributionsInference.extra_logprior(
    PooledPair(0.2, -0.1, 0.0), PooledPair(0.2, -0.1, 0.5), [0.5])
```

# See also
- [`logdensity`](@ref): adds this term after the per-row priors.
- [`parameter_rows`](@ref): the row schema this keeps to four fields.
"
extra_logprior(obj, reconstructed, x) = 0.0

@doc "

Reconstruct `obj` at `x` and score its [`extra_logprior`](@ref) together.

`reconstruct_with_logprior(obj, x)` returns `(reconstructed, extra)`, i.e.
[`reconstruct`](@ref)`(obj, x)` paired with
[`extra_logprior`](@ref)`(obj, reconstructed, x)`. The generic default below
is exactly that: call `reconstruct`, then `extra_logprior` on the result — no
behaviour change from calling the two separately.

This is an OPT-IN fusion point, not a new protocol requirement: it exists
because `reconstruct` and `extra_logprior` can each need the same expensive
intermediate representation of `x` (e.g. a nested-parameter unflattening) with
no way to share it when the engine calls them as two independent functions.
A type whose `reconstruct`/`extra_logprior` pair recomputes shared state can
override `reconstruct_with_logprior` directly to compute that state once and
reuse it for both halves, without changing the individual `reconstruct`/
`extra_logprior` methods (which stay correct, and are still what a caller
using either function on its own gets). See
`DistributionsInferenceComposedDistributionsExt` for the motivating case
(ComposedDistributions#212's centred-pooling internals).

# Arguments
- `obj`: the template fittable object.
- `x`: an estimated flat parameter vector of length
  [`flat_dimension`](@ref)`(obj)`.

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
DistributionsInference.reconstruct_with_logprior(leaf, [2.5])
```

# See also
- [`reconstruct`](@ref), [`extra_logprior`](@ref): the pair this fuses.
- [`logdensity`](@ref): the engine call site.
"
function reconstruct_with_logprior(obj, x::AbstractVector)
    reconstructed = reconstruct(obj, x)
    return reconstructed, extra_logprior(obj, reconstructed, x)
end
