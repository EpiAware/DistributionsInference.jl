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

# The estimated flat vector read from `chain`: a single `draw`'s values, or
# each row's draws reduced by `summary` over the `draws` selection. Mirrors
# ComposedDistributions' `chain_to_params` reduction semantics.
function _flat_from_chain(obj, chain; draw, draws, summary)
    rows = estimated_rows(obj)
    isempty(rows) && return Float64[]
    draw !== nothing &&
        return [vec(_chain_column(chain, row.name))[draw] for row in rows]
    sel = _draw_indices(chain, draws)
    return [summary(_select_draws(_chain_column(chain, row.name), sel)) for row in rows]
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
- [`to_flexichain`](@ref): build the chain this reads.
- [`readback_draws`](@ref): the vectorised, every-draw form.
"
function readback(obj, chain::FlexiChains.FlexiChain; summary = mean,
        draw = nothing, draws = nothing)
    x = _flat_from_chain(obj, chain; draw = draw, draws = draws, summary = summary)
    return reconstruct(obj, x)
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

# See also
- [`readback`](@ref): the single-draw / reduced read this vectorises.
- [`to_flexichain`](@ref): build the chain this reads.
"
function readback_draws(obj, chain::FlexiChains.FlexiChain; draws = nothing)
    sel = _draw_indices(chain, draws)
    idx = sel isa Colon ? (1:FlexiChains.niters(chain)) : sel
    return [readback(obj, chain; draw = i) for i in idx]
end
