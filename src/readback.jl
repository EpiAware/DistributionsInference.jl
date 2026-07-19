# Dotted-name FlexiChains readback: build a FlexiChain from the raw draws any
# LogDensityProblems-compatible sampler hands back, keyed by the estimated
# parameter rows' dotted names, and read it back onto a fitted object. No PPL
# is involved anywhere in this file — `FlexiChains` is a hard dependency of
# this package, so the naming contract this generalises (ComposedDistributions'
# `chain_to_params`/`param_draws`, ComposedDistributions#185) needs no glue
# extension here.

# Normalise raw sampler draws to a `dim x niter` matrix, checked against the
# object's estimated dimension. A LogDensityProblems-compatible sampler hands
# draws back as either shape: a `dim x niter` matrix (e.g. stacked HMC
# momenta), or a `niter`-length vector of `dim`-length vectors (e.g. AdvancedMH
# `Transition.params` collected over iterations).
function _draws_matrix(draws::AbstractMatrix, dim::Int)
    size(draws, 1) == dim || throw(DimensionMismatch(
        "draws matrix has $(size(draws, 1)) row(s) but $dim parameter(s) " *
        "are estimated"))
    return draws
end

function _draws_matrix(draws::AbstractVector{<:AbstractVector}, dim::Int)
    n = length(draws)
    T = n == 0 ? Float64 : eltype(first(draws))
    mat = Matrix{T}(undef, dim, n)
    for (j, d) in enumerate(draws)
        length(d) == dim || throw(DimensionMismatch(
            "draw $j has length $(length(d)) but $dim parameter(s) are " *
            "estimated"))
        mat[:, j] = d
    end
    return mat
end

function _draws_matrix(draws, dim::Int)
    throw(ArgumentError(
        "draws must be an AbstractMatrix (dim x niter) or an " *
        "AbstractVector of AbstractVectors (niter draws of dim-length " *
        "vectors); got $(typeof(draws))"))
end

@doc "

Build a dotted-name `FlexiChain` from raw sampler draws.

`to_flexichain(obj, draws)` keys `draws` by [`estimated_rows`](@ref)`(obj)`'s
dotted `name`s (in [`parameter_rows`](@ref) order), so the result reads back
onto `obj` with [`readback`](@ref)/[`readback_draws`](@ref). `draws` is
accepted in either raw shape a `LogDensityProblems`-compatible sampler hands
back: a `dim x niter` matrix, or a `niter`-length vector of `dim`-length
vectors, where `dim` is [`flat_dimension`](@ref)`(obj)`. An object estimating
nothing (`dim == 0`) still needs `draws` to carry the draw count — pass a
`(0, niter)` matrix or a `niter`-length vector of empty vectors.

No `DynamicPPL`/`Turing` involvement: this works with the draws of ANY sampler
that consumes [`as_logdensity`](@ref)`(obj, data)` through the
`LogDensityProblems` interface.

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `draws`: the raw draws, `dim x niter` or a `niter`-vector of `dim`-vectors.

# Examples
```@example
using DistributionsInference, Distributions

struct FlexiLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::FlexiLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

leaf = FlexiLeaf(2.0, 1.0)
draws = [2.1, 2.4, 2.0, 2.6]  # 1 estimated parameter, 4 draws
chain = DistributionsInference.to_flexichain(leaf, reshape(draws, 1, :))
using FlexiChains: FlexiChains
FlexiChains.parameters(chain)
```

# See also
- [`readback`](@ref): reduce the chain back onto `obj` (point summary/draw).
- [`readback_draws`](@ref): the vectorised, every-draw form.
"
function to_flexichain(obj, draws)
    rows = estimated_rows(obj)
    dim = length(rows)
    mat = _draws_matrix(draws, dim)
    niter = size(mat, 2)
    data = Dict{FlexiChains.ParameterOrExtra{<:Symbol}, Matrix}()
    for i in eachindex(rows)
        data[FlexiChains.Parameter(rows[i].name)] = reshape(mat[i, :], niter, 1)
    end
    return FlexiChains.FlexiChain{Symbol}(niter, 1, data)
end

# The named column for one estimated row, erroring with the dotted name (not
# a bare KeyError) when `chain` does not carry it — signals a chain that was
# not built (via `to_flexichain`) against this `obj`.
function _chain_column(chain, name::Symbol)
    FlexiChains.has_parameter(chain, name) ||
        throw(ArgumentError("parameter $(repr(name)) not found in chain"))
    return chain[name]
end

# The iteration indices a `draws` selector picks out, mirroring
# ComposedDistributions' `_draw_indices`: `nothing` is every iteration; a
# predicate filters the index range; anything else (a range / index vector) is
# taken as the indices directly.
_draw_indices(chain, ::Nothing) = Colon()
function _draw_indices(chain, draws)
    draws isa Function && return [i for i in 1:FlexiChains.niters(chain) if draws(i)]
    return collect(draws)
end

_select_draws(col, ::Colon) = vec(col)
_select_draws(col, sel) = vec(col)[sel]

@doc "

Read a dotted-name `FlexiChain`'s parameter values, keyed by name.

`distribution_params(obj, chain)` is the params-first readback primitive
(CD#195/DI#20): the estimated parameter values read from `chain`, keyed by
each [`estimated_rows`](@ref)`(obj)` row's dotted `name`, *before* any object
is rebuilt — a single `draw`'s values, or each row's draws reduced by
`summary` over the `draws` selection (default: the mean over every draw).
[`readback`](@ref) is a thin layer on top: it collapses this result to a flat
vector (`estimated_rows` order is fixed, so `values(...)` recovers it) and
calls [`reconstruct`](@ref).

The argument order is `obj` first, `chain` second — matching
`to_flexichain(obj, draws)` and `readback(obj, chain)` in this same file, and
ComposedDistributions' `chain_to_params(template, chain)` (the function this
generalises, CD#195/DI#20): keeping one order across the module avoids a
silent argument swap between sibling calls.

# Arguments
- `obj`: the fittable object the chain's parameters were sampled for.
- `chain`: the `FlexiChain` to read parameter values from (see
  [`to_flexichain`](@ref)).

# Keyword Arguments
- `summary`: the reduction `AbstractVector -> scalar` applied to each row's
  draws (default `mean`); ignored when `draw` is given.
- `draw`: a single iteration index to read, overriding `summary`/`draws`.
- `draws`: a subset of iterations to reduce over (a range / index vector, or a
  predicate over the iteration index); `nothing` uses every iteration.

Two estimated rows sharing a dotted `name` is refused with a clear
`ArgumentError` naming the duplicate: a `NamedTuple` cannot key two entries
by the same name, and a repeated name can only mean `parameter_rows(obj)`
gave two distinct parameters the same identifier (a protocol bug in `obj`'s
own implementation), not a case with a sensible silent resolution.

# Examples
```@example
using DistributionsInference, Distributions

struct ParamsLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::ParamsLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

leaf = ParamsLeaf(2.0, 1.0)
draws = [2.1, 2.4, 2.0, 2.6]
chain = DistributionsInference.to_flexichain(leaf, reshape(draws, 1, :))
DistributionsInference.distribution_params(leaf, chain)
```

# See also
- [`readback`](@ref): rebuilds `obj` from this primitive's result.
- [`readback_draws`](@ref): the vectorised, every-draw form (its own
  optimised implementation, not layered on this — see its docstring).
"
function distribution_params(obj, chain::FlexiChains.FlexiChain;
        summary = mean, draw = nothing, draws = nothing)
    rows = estimated_rows(obj)
    isempty(rows) && return NamedTuple()
    names = Tuple(row.name for row in rows)
    _check_unique_names(names)
    vals = if draw !== nothing
        [vec(_chain_column(chain, row.name))[draw] for row in rows]
    else
        sel = _draw_indices(chain, draws)
        [summary(_select_draws(_chain_column(chain, row.name), sel))
         for row in rows]
    end
    return NamedTuple{names}(Tuple(vals))
end

# `NamedTuple{names}(...)` fails on a repeated name with a bare "duplicate
# field name" error that does not say which object or row is at fault. A
# duplicate can only come from a `parameter_rows(obj)` implementation that
# gives two estimated rows the same dotted `name` (a protocol bug, not a
# normal case: every row is meant to name one distinct parameter) — refuse
# it here, at the earliest point the duplicate is visible, with a message
# that names the object and the repeated name(s), rather than let it surface
# later as a puzzling `NamedTuple` construction error.
function _check_unique_names(names::Tuple)
    length(names) == length(Set(names)) && return nothing
    counts = Dict{Symbol, Int}()
    for n in names
        counts[n] = get(counts, n, 0) + 1
    end
    dupes = [n for (n, c) in counts if c > 1]
    throw(ArgumentError(
        "distribution_params: duplicate estimated parameter name(s) " *
        "$(dupes); parameter_rows(obj) must give every estimated row a " *
        "unique dotted name"))
end

@doc "

Read a dotted-name `FlexiChain` back onto a fitted object.

`readback(obj, chain)` reduces `chain` (built by [`to_flexichain`](@ref)) to a
flat estimated parameter vector and rebuilds a concrete object via
[`reconstruct`](@ref): a point summary by default (`summary` applied to each
estimated row's draws, default `mean`), a single iteration (`draw`), or a
summary restricted to a subset of iterations (`draws`).

# Arguments
- `obj`: the fittable object the chain's parameters were sampled for.
- `chain`: the `FlexiChain` to read parameter values from (see
  [`to_flexichain`](@ref)).

# Keyword Arguments
- `summary`: the reduction `AbstractVector -> scalar` applied to each row's
  draws (default `mean`); ignored when `draw` is given.
- `draw`: a single iteration index to read, overriding `summary`/`draws`.
- `draws`: a subset of iterations to reduce over (a range / index vector, or a
  predicate over the iteration index); `nothing` uses every iteration.

# Examples
```@example
using DistributionsInference, Distributions

struct ReadbackLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::ReadbackLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(d::ReadbackLeaf, x::AbstractVector)
    return ReadbackLeaf(x[1], d.scale)
end

leaf = ReadbackLeaf(2.0, 1.0)
draws = [2.1, 2.4, 2.0, 2.6]
chain = DistributionsInference.to_flexichain(leaf, reshape(draws, 1, :))
DistributionsInference.readback(leaf, chain).shape
```

# See also
- [`distribution_params`](@ref): the params-first primitive this layers on.
- [`to_flexichain`](@ref): build the chain this reads.
- [`readback_draws`](@ref): the vectorised, every-draw form.
"
function readback(obj, chain::FlexiChains.FlexiChain; summary = mean,
        draw = nothing, draws = nothing)
    nt = distribution_params(obj, chain; summary = summary, draw = draw,
        draws = draws)
    return reconstruct(obj, collect(values(nt)))
end

@doc "

Read every draw of a dotted-name `FlexiChain` back onto a fitted object.

`readback_draws(obj, chain)` is the vectorised form of [`readback`](@ref):
where `readback` reduces the chain to one reconstructed object,
`readback_draws` keeps every draw, returning a vector of reconstructed
objects (one per selected iteration) — e.g. for a per-draw
posterior-predictive summary.

# Arguments
- `obj`: the fittable object the chain's parameters were sampled for.
- `chain`: the `FlexiChain` to read every draw from (see
  [`to_flexichain`](@ref)).

# Keyword Arguments
- `draws`: a subset of iterations to keep (a range / index vector, or a
  predicate over the iteration index); `nothing` keeps every iteration.

# Examples
```@example
using DistributionsInference, Distributions

struct DrawsLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::DrawsLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(d::DrawsLeaf, x::AbstractVector)
    return DrawsLeaf(x[1], d.scale)
end

leaf = DrawsLeaf(2.0, 1.0)
draws = [2.1, 2.4, 2.0, 2.6]
chain = DistributionsInference.to_flexichain(leaf, reshape(draws, 1, :))
length(DistributionsInference.readback_draws(leaf, chain))
```

!!! note \"Not layered on `distribution_params`\"
    Unlike [`readback`](@ref), this does *not* call
    [`distribution_params`](@ref) once per draw: `distribution_params`
    re-fetches and re-validates every estimated row's column on each call,
    which would be O(niter x nrows) column look-ups here instead of the
    O(nrows) this implementation does by materialising each column once
    up front. The two stay independent implementations of the same
    per-draw extraction for this reason.

# See also
- [`readback`](@ref): the single-draw / reduced read this vectorises.
- [`distribution_params`](@ref): the params-first primitive `readback` (but
  not this function) layers on.
- [`to_flexichain`](@ref): build the chain this reads.
"
function readback_draws(obj, chain::FlexiChains.FlexiChain; draws = nothing)
    rows = estimated_rows(obj)
    sel = _draw_indices(chain, draws)
    idx = sel isa Colon ? (1:FlexiChains.niters(chain)) : sel
    isempty(rows) && return [reconstruct(obj, Float64[]) for _ in idx]
    # Materialise each estimated row's column once, then index per draw, so
    # this stays O(niter) rather than re-extracting every column per draw.
    cols = [vec(_chain_column(chain, row.name)) for row in rows]
    return [reconstruct(obj, [col[i] for col in cols]) for i in idx]
end
