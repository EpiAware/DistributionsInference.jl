# Dotted-name FlexiChains readback: build a FlexiChain from the raw draws any
# LogDensityProblems-compatible sampler hands back, keyed by the estimated
# parameter rows' dotted names. `FlexiChains` is a hard dependency, so
# `draws_to_chain` and its helpers are implemented directly here; the
# `VarName`-keyed dispatch (a chain sampled from `distribution_to_turing`)
# lives in the `DistributionsInferenceDynamicPPLFlexiChainsExt` extension,
# which needs `DynamicPPL` loaded alongside this package. The chain-to-object
# readback itself (`inference_to_distribution`/`inference_to_distributions`)
# is in `src/inference.jl`, and shares `_chain_column`/`_pooled_ndraws`/
# `_reconstruct_pooled`/`_wrong_chain_type` from here.
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
[`inference_to_distribution`](@ref)/[`inference_to_distributions`](@ref).
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
- [`inference_to_distribution`](@ref): the Monte Carlo posterior predictive
  over a chain (or these raw draws directly).
- [`inference_to_distributions`](@ref): the vectorised, every-draw form.
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

# Rebuild one object per pooled draw index in `idx`, shared by
# `inference_to_distributions` (and, through it, `inference_to_distribution`).
# Materialises each estimated row's column once, so this stays O(niter)
# rather than re-extracting every column per draw.
function _reconstruct_pooled(obj, chain::FlexiChains.FlexiChain, idx)
    rows = estimated_rows(obj)
    isempty(rows) && return [reconstruct(obj, Float64[]) for _ in idx]
    cols = [vec(_chain_column(chain, row.name)) for row in rows]
    return [reconstruct(obj, [col[i] for col in cols]) for i in idx]
end
