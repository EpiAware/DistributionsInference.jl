# Dotted-name FlexiChains readback: build a FlexiChain from the raw draws any
# LogDensityProblems-compatible sampler hands back, keyed by the estimated
# parameter rows' dotted names, and read it back onto a fitted object.
# `FlexiChains` is a hard dependency, so all four functions are implemented
# directly here; only the `VarName`-keyed dispatch (a chain sampled from
# `distribution_to_turing`) lives in the `DistributionsInferenceDynamicPPLFlexiChainsExt`
# extension, which needs `DynamicPPL` loaded alongside this package.
#
# `draws_to_chain` (below) is `public`: an engine whose sampler hands back raw
# draws rather than a chain of its own needs it to satisfy
# `distribution_to_chain`'s `FlexiChain`-returning contract (#94).

# Normalise raw sampler draws to a `dim x niter` matrix, checked against the
# object's estimated dimension. Samplers hand draws back either as a
# `dim x niter` matrix or as a `niter`-length vector of `dim`-length vectors.
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

_draws_matrix(draws, dim::Int) = _malformed_draws(draws)

function _malformed_draws(draws)
    throw(ArgumentError(
        "draws must be an AbstractMatrix (dim x niter) or an " *
        "AbstractVector of AbstractVectors (niter draws of dim-length " *
        "vectors); got $(typeof(draws))"))
end

# The fallback for a chain argument no typed method matched: with
# `FlexiChains` always in the session, that can only be the wrong chain type.
function _wrong_chain_type(f::Symbol, chain)
    throw(ArgumentError(
        "`$f` has no method for a chain of type $(typeof(chain)): it reads " *
        "a `FlexiChains.FlexiChain` keyed by the estimated rows' dotted " *
        "names, or a `VarName`-keyed chain once `DynamicPPL` is loaded " *
        "alongside it."))
end

@doc "

Build a dotted-name `FlexiChain` from raw sampler draws.

`draws_to_chain(obj, draws; nchains = 1)` keys `draws` by
[`estimated_rows`](@ref)`(obj)`'s dotted `name`s (in [`parameter_rows`](@ref)
order), so the result reads back onto `obj` with
[`point_estimate`](@ref)/[`distribution_draws`](@ref)/[`inference_to_distribution`](@ref).
`draws` is accepted in either raw shape a `LogDensityProblems`-compatible
sampler hands back: a `dim x niter` matrix, or a `niter`-length vector of
`dim`-length vectors, where `dim` is [`flat_dimension`](@ref)`(obj)`. An
object estimating nothing (`dim == 0`) still needs `draws` to carry the draw
count — pass a `(0, niter)` matrix or a `niter`-length vector of empty
vectors.

`nchains` splits the pooled `niter` draws into that many equal chains,
chain-major: the first `niter / nchains` columns are chain 1, the next
`niter / nchains` are chain 2, and so on. `niter` must be an exact multiple
of `nchains`, checked here — this is the one place that can catch a chain
count and a draw count silently disagreeing (#89). Default `nchains = 1`
treats every column as one chain.

`public`, not exported: an ordinary caller reaches
[`inference_to_distribution`](@ref)/[`inference_to_distributions`](@ref)
directly on raw draws instead, which build this internally. This function is
for an engine author whose sampler hands back raw draws rather than a chain
of its own (e.g. a `LogDensityProblems`-driven sampler) and needs to satisfy
[`distribution_to_chain`](@ref)'s `FlexiChain`-returning contract (#94).

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `draws`: the raw draws, `dim x niter` or a `niter`-vector of `dim`-vectors.

# Keyword Arguments
- `nchains`: the number of chains the pooled columns split into, chain-major
  (default `1`).

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChains

struct ChainLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::ChainLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

leaf = ChainLeaf(2.0, 1.0)
draws = reshape([2.1, 2.4, 2.0, 2.6], 1, 4)
chain = DistributionsInference.draws_to_chain(leaf, draws)
FlexiChains.has_parameter(chain, :shape)
```

# See also
- [`point_estimate`](@ref): reduce a chain back onto `obj` (point
  summary/draw).
- [`distribution_draws`](@ref): the vectorised, every-draw form.
- [`inference_to_distribution`](@ref): the Monte Carlo posterior predictive
  over a chain (or these raw draws directly).
- [`distribution_to_chain`](@ref): the engine contract this satisfies.
"
function draws_to_chain(obj, draws; nchains::Int = 1)
    nchains >= 1 || throw(ArgumentError(
        "draws_to_chain: nchains must be >= 1, got $nchains"))
    rows = estimated_rows(obj)
    dim = length(rows)
    mat = _draws_matrix(draws, dim)
    total = size(mat, 2)
    niters, remainder = divrem(total, nchains)
    remainder == 0 || throw(ArgumentError(
        "draws_to_chain: $total pooled draw(s) is not an exact multiple " *
        "of nchains=$nchains; pass a draws count divisible by nchains, or " *
        "omit nchains to treat every column as one chain"))
    data = Dict{FlexiChains.ParameterOrExtra{<:Symbol}, Matrix}()
    for i in eachindex(rows)
        data[FlexiChains.Parameter(rows[i].name)] = reshape(
            mat[i, :], niters, nchains)
    end
    return FlexiChains.FlexiChain{Symbol}(niters, nchains, data)
end

function _chain_column(chain::FlexiChains.FlexiChain, name::Symbol)
    FlexiChains.has_parameter(chain, name) ||
        throw(ArgumentError("parameter $(repr(name)) not found in chain"))
    return chain[name]
end

# The number of draws pooled across every chain: `_chain_column` returns the
# full niters x nchains array and `vec` flattens it column-major, so the
# pooled range a draw selector or an index into it must span is
# niters * nchains, not niters alone (one chain's worth).
function _pooled_ndraws(chain::FlexiChains.FlexiChain)
    FlexiChains.niters(chain) * FlexiChains.nchains(chain)
end

# The pooled-draw indices a `draws` selector picks out: `nothing` is every
# draw, a predicate filters the pooled index range, anything else is taken
# as the indices directly.
_draw_indices(chain, ::Nothing) = Colon()
function _draw_indices(chain, draws)
    draws isa Function &&
        return [i for i in 1:_pooled_ndraws(chain) if draws(i)]
    return collect(draws)
end

_select_draws(col, ::Colon) = vec(col)
_select_draws(col, sel) = vec(col)[sel]

# `NamedTuple{names}(...)` fails on a repeated name with a bare "duplicate
# field name" error that names neither the object nor the row, so catch it at
# the earliest point the duplicate is visible.
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

Read a dotted-name `FlexiChain`'s parameter values, keyed by name.

`distribution_params(obj, chain)` is the params-first readback primitive: the
estimated parameter values read from `chain`, keyed by each
[`estimated_rows`](@ref)`(obj)` row's dotted `name`, *before* any object is
rebuilt — a single `draw`'s values, or each row's draws reduced by `summary`
over the `draws` selection (default: the mean over every draw).
[`point_estimate`](@ref) is a thin layer on top: it collapses this result to a
flat vector and calls [`reconstruct`](@ref).

A `VarName`-keyed chain (one sampled from [`distribution_to_turing`](@ref))
is read by the `DistributionsInferenceDynamicPPLFlexiChainsExt` extension,
which needs `DynamicPPL` loaded as well.

# Arguments
- `obj`: the fittable object the chain's parameters were sampled for.
- `chain`: the `FlexiChain` to read parameter values from, keyed by the
  estimated rows' dotted names.

# Keyword Arguments
- `summary`: the reduction `AbstractVector -> scalar` applied to each row's
  draws (default `mean`); ignored when `draw` is given.
- `draw`: a single iteration index to read, overriding `summary`/`draws`.
- `draws`: a subset of iterations to reduce over (a range / index vector, or a
  predicate over the iteration index); `nothing` uses every iteration.

Two estimated rows sharing a dotted `name` is refused with an `ArgumentError`
naming the duplicate: a `NamedTuple` cannot key two entries by the same name,
and a repeated name means a protocol bug in `obj`'s `parameter_rows`.

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChain, Parameter

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
chain = FlexiChain{Symbol}(
    4, 1, Dict(Parameter(:shape) => reshape(draws, 4, 1)))
distribution_params(leaf, chain)
```

# See also
- [`point_estimate`](@ref): rebuilds `obj` from this primitive's result.
- [`distribution_draws`](@ref): the vectorised, every-draw form (its own
  optimised implementation, not layered on this — see its docstring).
"
function distribution_params(obj, chain; kwargs...)
    return _wrong_chain_type(:distribution_params, chain)
end

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

@doc "

Read a dotted-name `FlexiChain` back onto a fitted object.

`point_estimate(obj, chain)` reduces `chain` to a flat estimated parameter
vector and rebuilds a concrete object via [`reconstruct`](@ref): a point
summary by default
(`summary` applied to each estimated row's draws, default `mean`), a single
iteration (`draw`), or a summary restricted to a subset of iterations
(`draws`).

A `VarName`-keyed chain (one sampled from [`distribution_to_turing`](@ref))
is read by the `DistributionsInferenceDynamicPPLFlexiChainsExt` extension,
which needs `DynamicPPL` loaded as well.

# Arguments
- `obj`: the fittable object the chain's parameters were sampled for.
- `chain`: the `FlexiChain` to read parameter values from, keyed by the
  estimated rows' dotted names.

# Keyword Arguments
- `summary`: the reduction `AbstractVector -> scalar` applied to each row's
  draws (default `mean`); ignored when `draw` is given.
- `draw`: a single iteration index to read, overriding `summary`/`draws`.
- `draws`: a subset of iterations to reduce over (a range / index vector, or a
  predicate over the iteration index); `nothing` uses every iteration.

!!! note \"Not the posterior predictive\"
    This is a plug-in estimate, reducing draws to a point *before* rebuilding
    `obj` — `Gamma(mean(shape draws), scale) != mean(Gamma.(shape draws,
    scale))`. For the Monte Carlo posterior predictive, see
    [`inference_to_distribution`](@ref).

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChain, Parameter

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
chain = FlexiChain{Symbol}(
    4, 1, Dict(Parameter(:shape) => reshape(draws, 4, 1)))
point_estimate(leaf, chain).shape
```

# See also
- [`distribution_params`](@ref): the params-first primitive this layers on.
- [`distribution_draws`](@ref): the vectorised, every-draw form.
- [`inference_to_distribution`](@ref): the Monte Carlo posterior predictive,
  the honest alternative to this plug-in reduction.
"
function point_estimate(obj, chain; kwargs...)
    return _wrong_chain_type(:point_estimate, chain)
end

function point_estimate(obj, chain::FlexiChains.FlexiChain; summary = mean,
        draw = nothing, draws = nothing)
    nt = distribution_params(obj, chain; summary = summary, draw = draw,
        draws = draws)
    return reconstruct(obj, collect(values(nt)))
end

@doc "

Read every draw of a dotted-name `FlexiChain` back onto a fitted object.

`distribution_draws(obj, chain)` is the vectorised form of
[`point_estimate`](@ref): where `point_estimate` reduces the chain to one
reconstructed object, `distribution_draws` keeps every draw, returning a
vector of reconstructed objects (one per selected iteration) — e.g. for a
per-draw posterior-predictive summary.

A `VarName`-keyed chain (one sampled from [`distribution_to_turing`](@ref))
is read by the `DistributionsInferenceDynamicPPLFlexiChainsExt` extension,
which needs `DynamicPPL` loaded as well.

# Arguments
- `obj`: the fittable object the chain's parameters were sampled for.
- `chain`: the `FlexiChain` to read every draw from, keyed by the estimated
  rows' dotted names.

# Keyword Arguments
- `draws`: a subset of iterations to keep (a range / index vector, or a
  predicate over the iteration index); `nothing` keeps every iteration.

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChain, Parameter

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
chain = FlexiChain{Symbol}(
    4, 1, Dict(Parameter(:shape) => reshape(draws, 4, 1)))
length(distribution_draws(leaf, chain))
```

!!! note \"Not layered on `distribution_params`\"
    Unlike [`point_estimate`](@ref), this does *not* call
    [`distribution_params`](@ref) once per draw, which would re-fetch and
    re-validate every estimated row's column on each call. It materialises
    each column once instead, so the two are independent implementations of
    the same per-draw extraction.

# See also
- [`point_estimate`](@ref): the single-draw / reduced read this vectorises.
- [`distribution_params`](@ref): the params-first primitive `point_estimate`
  (but not this function) layers on.
- [`inference_to_distributions`](@ref): the additive replacement for this
  function, with the chain-major-pooling trap documented and a random
  `draws::Integer` subsample form.
"
function distribution_draws(obj, chain; kwargs...)
    return _wrong_chain_type(:distribution_draws, chain)
end

function distribution_draws(obj, chain::FlexiChains.FlexiChain; draws = nothing)
    sel = _draw_indices(chain, draws)
    idx = sel isa Colon ? (1:_pooled_ndraws(chain)) : sel
    return _reconstruct_pooled(obj, chain, idx)
end

# Shared by `distribution_draws` and `inference_to_distributions`: rebuild one
# object per pooled draw index in `idx`. Materialises each estimated row's
# column once, so this stays O(niter) rather than re-extracting every column
# per draw — the two callers differ only in how `idx` is computed
# (`_draw_indices`'s predicate/range selector vs `_resolve_pooled_draws`'s
# four `draws` forms), not in this extraction.
function _reconstruct_pooled(obj, chain::FlexiChains.FlexiChain, idx)
    rows = estimated_rows(obj)
    isempty(rows) && return [reconstruct(obj, Float64[]) for _ in idx]
    cols = [vec(_chain_column(chain, row.name)) for row in rows]
    return [reconstruct(obj, [col[i] for col in cols]) for i in idx]
end
