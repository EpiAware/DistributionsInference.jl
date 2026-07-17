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
  parameter, e.g. `:onset.shape`).
- `value`: the parameter's current value.
- `prior`: the attached prior (a `UnivariateDistribution`) if the parameter is
  ESTIMATED, or `nothing` if it is fixed at `value`.
- `support`: the `(lower, upper)` bounds of the parameter's admissible domain.

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
