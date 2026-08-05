# Dotted-name FlexiChains readback: build a FlexiChain from the raw draws any
# LogDensityProblems-compatible sampler hands back, keyed by the estimated
# parameter rows' dotted names, and read it back onto a fitted object. The
# four functions are declared here as FlexiChains-free stubs carrying the
# docstrings, and implemented in `DistributionsInferenceFlexiChainsExt`.

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

# Whether the readback extension is live in this session, which separates "you
# have not loaded FlexiChains" from "you passed the wrong chain type".
function _flexichains_loaded()
    return Base.get_extension(
        @__MODULE__, :DistributionsInferenceFlexiChainsExt) !== nothing
end

# Whether `FlexiChains` itself is in the session, a different question from
# whether the extension loaded: `Base.get_extension` returns `nothing` both
# when the trigger package was never loaded and when the extension failed to
# load. Telling those apart stops the error below advising `using FlexiChains`
# to someone who already ran it.
const _FLEXICHAINS_PKGID = Base.PkgId(
    Base.UUID("4a37a8b9-6e57-4b92-8664-298d46e639f7"), "FlexiChains")

_flexichains_present() = haskey(Base.loaded_modules, _FLEXICHAINS_PKGID)

function _flexichains_required(f::Symbol)
    _flexichains_present() && throw(ArgumentError(
        "`$f` has no method: `FlexiChains` is loaded, but the " *
        "`DistributionsInferenceFlexiChainsExt` package extension carrying " *
        "the dotted-name chain readback is not in this session, so it " *
        "failed to load. Check this session's precompilation warnings for " *
        "that extension, and the installed `FlexiChains` version against " *
        "this package's `[compat]` bound."))
    throw(ArgumentError(
        "`$f` needs `FlexiChains`: the dotted-name chain readback lives in " *
        "the `DistributionsInferenceFlexiChainsExt` package extension, " *
        "which loads only once `FlexiChains` is in the session. Run " *
        "`using FlexiChains` first (and `Pkg.add(\"FlexiChains\")` if it is " *
        "not installed yet)."))
end

# The fallback for a chain argument the extension's own methods did not match:
# with `FlexiChains` loaded that can only be the wrong chain type.
function _no_chain_method(f::Symbol, chain)
    _flexichains_loaded() || _flexichains_required(f)
    throw(ArgumentError(
        "`$f` has no method for a chain of type $(typeof(chain)): it reads " *
        "a `FlexiChains.FlexiChain` keyed by the estimated rows' dotted " *
        "names, or a `VarName`-keyed chain once `DynamicPPL` is loaded " *
        "alongside `FlexiChains`."))
end

@doc "

Build a dotted-name `FlexiChain` from raw sampler draws. Internal.

`_to_flexichain(obj, draws)` keys `draws` by [`estimated_rows`](@ref)`(obj)`'s
dotted `name`s (in [`parameter_rows`](@ref) order), so the result reads back
onto `obj` with [`point_estimate`](@ref)/[`distribution_draws`](@ref). `draws`
is accepted in either raw shape a `LogDensityProblems`-compatible sampler hands
back: a `dim x niter` matrix, or a `niter`-length vector of `dim`-length
vectors, where `dim` is [`flat_dimension`](@ref)`(obj)`. An object estimating
nothing (`dim == 0`) still needs `draws` to carry the draw count — pass a
`(0, niter)` matrix or a `niter`-length vector of empty vectors.

Not part of the public surface: standardising a sampler's raw draws into a
chain type belongs to `FlexiChains` or to the inference package that produced
them, not here (#91 takes the conversion off the readback path entirely).

This has no method until `FlexiChains` is loaded; the chain construction lives
in the `DistributionsInferenceFlexiChainsExt` extension.

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `draws`: the raw draws, `dim x niter` or a `niter`-vector of `dim`-vectors.

# See also
- [`point_estimate`](@ref): reduce a chain back onto `obj` (point
  summary/draw).
- [`distribution_draws`](@ref): the vectorised, every-draw form.
"
function _to_flexichain(obj, draws)
    _flexichains_loaded() || _flexichains_required(:_to_flexichain)
    # With the extension loaded, either accepted raw shape would have matched
    # its typed methods, so reaching here means the shape is wrong.
    return _malformed_draws(draws)
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

This has no method until `FlexiChains` is loaded; the read lives in the
`DistributionsInferenceFlexiChainsExt` extension. A `VarName`-keyed chain (one
sampled from [`distribution_to_turing`](@ref)) is read by the
`DistributionsInferenceDynamicPPLFlexiChainsExt` extension, which needs
`DynamicPPL` loaded as well.

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
    return _no_chain_method(:distribution_params, chain)
end

@doc "

Read a dotted-name `FlexiChain` back onto a fitted object.

`point_estimate(obj, chain)` reduces `chain` to a flat estimated parameter
vector and rebuilds a concrete object via [`reconstruct`](@ref): a point
summary by default
(`summary` applied to each estimated row's draws, default `mean`), a single
iteration (`draw`), or a summary restricted to a subset of iterations
(`draws`).

This has no method until `FlexiChains` is loaded; the read lives in the
`DistributionsInferenceFlexiChainsExt` extension. A `VarName`-keyed chain (one
sampled from [`distribution_to_turing`](@ref)) is read by the
`DistributionsInferenceDynamicPPLFlexiChainsExt` extension, which needs
`DynamicPPL` loaded as well.

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
"
function point_estimate(obj, chain; kwargs...)
    return _no_chain_method(:point_estimate, chain)
end

@doc "

Read every draw of a dotted-name `FlexiChain` back onto a fitted object.

`distribution_draws(obj, chain)` is the vectorised form of
[`point_estimate`](@ref): where `point_estimate` reduces the chain to one
reconstructed object, `distribution_draws` keeps every draw, returning a
vector of reconstructed objects (one per selected iteration) — e.g. for a
per-draw posterior-predictive summary.

This has no method until `FlexiChains` is loaded; the read lives in the
`DistributionsInferenceFlexiChainsExt` extension. A `VarName`-keyed chain (one
sampled from [`distribution_to_turing`](@ref)) is read by the
`DistributionsInferenceDynamicPPLFlexiChainsExt` extension, which needs
`DynamicPPL` loaded as well.

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
"
function distribution_draws(obj, chain; kwargs...)
    return _no_chain_method(:distribution_draws, chain)
end
