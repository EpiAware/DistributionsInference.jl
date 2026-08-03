# Dotted-name FlexiChains readback: build a FlexiChain from the raw draws any
# LogDensityProblems-compatible sampler hands back, keyed by the estimated
# parameter rows' dotted names, and read it back onto a fitted object. No PPL
# is involved anywhere (the naming contract this generalises,
# ComposedDistributions' `chain_to_params`/`param_draws`,
# ComposedDistributions#185, needs no glue extension), but the chain type
# itself comes from `FlexiChains`, so the four functions are declared here as
# FlexiChains-free stubs (with their docstrings, the single source of truth)
# and implemented in the weakdep `DistributionsInferenceFlexiChainsExt`
# extension (`ext/`), loaded only when `FlexiChains` is present. The draw-shape
# validation the readback shares (`_draws_matrix`) mentions no chain type, so
# it stays here.

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

_draws_matrix(draws, dim::Int) = _malformed_draws(draws)

# `draws` in neither raw shape. Shared with `to_flexichain`'s own stub below,
# which reports it for a `draws` the extension's methods never see.
function _malformed_draws(draws)
    throw(ArgumentError(
        "draws must be an AbstractMatrix (dim x niter) or an " *
        "AbstractVector of AbstractVectors (niter draws of dim-length " *
        "vectors); got $(typeof(draws))"))
end

# Whether the readback extension is live in this session. Every method of the
# four functions below lives there, so this is what separates "you have not
# loaded FlexiChains" from "you passed the wrong chain type".
function _flexichains_loaded()
    return Base.get_extension(
        @__MODULE__, :DistributionsInferenceFlexiChainsExt) !== nothing
end

# Whether `FlexiChains` itself is in the session, which is a different question
# from whether the extension loaded: `Base.get_extension` returns `nothing`
# both when the trigger package was never loaded and when the extension failed
# to precompile or load, and the latter is reported only as a warning that is
# easy to miss. Telling those apart is what stops the error below advising
# `using FlexiChains` to someone who already ran it.
const _FLEXICHAINS_PKGID = Base.PkgId(
    Base.UUID("4a37a8b9-6e57-4b92-8664-298d46e639f7"), "FlexiChains")

_flexichains_present() = haskey(Base.loaded_modules, _FLEXICHAINS_PKGID)

# A readback call made while the extension carrying every method is absent. The
# call would otherwise fail with a bare `MethodError` naming neither the
# package nor the extension, so name both and the fix — and, when `FlexiChains`
# is already loaded, point at the extension's own load failure instead.
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

# The fallback for a chain argument the extension's own methods did not match.
# With `FlexiChains` loaded that can only be the wrong chain type, so say so
# rather than blaming the (present) package.
function _no_chain_method(f::Symbol, chain)
    _flexichains_loaded() || _flexichains_required(f)
    throw(ArgumentError(
        "`$f` has no method for a chain of type $(typeof(chain)): it reads " *
        "a `FlexiChains.FlexiChain` keyed by the estimated rows' dotted " *
        "names (build one with `to_flexichain`), or a `VarName`-keyed chain " *
        "once `DynamicPPL` is loaded alongside `FlexiChains`."))
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

This has no method until `FlexiChains` is loaded; the chain construction lives
in the `DistributionsInferenceFlexiChainsExt` extension, so the core package
stays free of a `FlexiChains` dependency. Calling it beforehand raises an
`ArgumentError` naming the package to load.

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `draws`: the raw draws, `dim x niter` or a `niter`-vector of `dim`-vectors.

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChains

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
FlexiChains.parameters(chain)
```

# See also
- [`readback`](@ref): reduce the chain back onto `obj` (point summary/draw).
- [`readback_draws`](@ref): the vectorised, every-draw form.
"
function to_flexichain(obj, draws)
    _flexichains_loaded() || _flexichains_required(:to_flexichain)
    # `FlexiChains` IS loaded, so a `draws` in either accepted raw shape would
    # have matched the extension's own (typed) methods: reaching this fallback
    # can only mean the shape is wrong.
    return _malformed_draws(draws)
end

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

This has no method until `FlexiChains` is loaded; the read lives in the
`DistributionsInferenceFlexiChainsExt` extension. A `VarName`-keyed chain (one
sampled from [`as_turing`](@ref)) is read by the
`DistributionsInferenceDynamicPPLFlexiChainsExt` extension, which needs
`DynamicPPL` loaded as well.

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
using FlexiChains: FlexiChains

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
function distribution_params(obj, chain; kwargs...)
    return _no_chain_method(:distribution_params, chain)
end

@doc "

Read a dotted-name `FlexiChain` back onto a fitted object.

`readback(obj, chain)` reduces `chain` (built by [`to_flexichain`](@ref)) to a
flat estimated parameter vector and rebuilds a concrete object via
[`reconstruct`](@ref): a point summary by default (`summary` applied to each
estimated row's draws, default `mean`), a single iteration (`draw`), or a
summary restricted to a subset of iterations (`draws`).

This has no method until `FlexiChains` is loaded; the read lives in the
`DistributionsInferenceFlexiChainsExt` extension. A `VarName`-keyed chain (one
sampled from [`as_turing`](@ref)) is read by the
`DistributionsInferenceDynamicPPLFlexiChainsExt` extension, which needs
`DynamicPPL` loaded as well.

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
using FlexiChains: FlexiChains

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
function readback(obj, chain; kwargs...)
    return _no_chain_method(:readback, chain)
end

@doc "

Read every draw of a dotted-name `FlexiChain` back onto a fitted object.

`readback_draws(obj, chain)` is the vectorised form of [`readback`](@ref):
where `readback` reduces the chain to one reconstructed object,
`readback_draws` keeps every draw, returning a vector of reconstructed
objects (one per selected iteration) — e.g. for a per-draw
posterior-predictive summary.

This has no method until `FlexiChains` is loaded; the read lives in the
`DistributionsInferenceFlexiChainsExt` extension. A `VarName`-keyed chain (one
sampled from [`as_turing`](@ref)) is read by the
`DistributionsInferenceDynamicPPLFlexiChainsExt` extension, which needs
`DynamicPPL` loaded as well.

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
using FlexiChains: FlexiChains

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
function readback_draws(obj, chain; kwargs...)
    return _no_chain_method(:readback_draws, chain)
end
