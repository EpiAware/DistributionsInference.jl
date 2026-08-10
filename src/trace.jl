# PosteriorTrace: the FlexiChains-free posterior representation at the core
# of the trace redesign (#90). A `dim x ndraws` matrix of pooled draws (chain
# boundaries are counted, not addressed), read back onto `obj` on demand via
# `reconstruct`. `stats` is reserved for per-draw extras (log-density,
# divergence flags, ...) an engine will populate under #94; nothing populates
# it yet, but the field ships now, since adding one later would be breaking
# (see the #90 sign-off).
#
# `T` is a phantom parameter (it names no field), so it cannot be inferred
# from arguments the way `O`/`D`/`S` can. There is exactly one path to a
# `PosteriorTrace`: the outer constructor below, which computes `T` itself
# from `reconstruct`'s return type and calls the struct's own `{T,O,D,S}`
# constructor explicitly. Both `draws_to_trace` and the slicing `getindex`
# route through it — neither constructs a `PosteriorTrace{...}` directly —
# so `T` is never left to (failed) inference. See the #90 sign-off's
# "adversarial review" correction for the bug this avoids: the issue's own
# centrepiece slicing example did not run under the naive
# `PosteriorTrace(obj, draws, names, nchains)` call the auto-generated inner
# constructor cannot resolve.

@doc "

A pooled posterior trace over a fittable object's estimated parameters.

`PosteriorTrace{T, O, D, S} <: AbstractVector{T}` holds the raw draws a
sampler produced for `obj`, without reconstructing an object per draw until
asked: `trace[i]` calls [`reconstruct`](@ref)`(obj, draws[:, i])` on demand,
so a trace is as cheap to hold as its draw matrix. `T` is fixed once, from
`reconstruct`'s return type on the first draw — which is also why an empty
trace (`ndraws == 0`) is refused at construction rather than given an
invented element type. `reconstruct` must return one concrete type per
`(obj, eltype(draws))` for `T` to mean anything: an implementation that
branches on a draw's runtime values to pick different concrete result types
would make later elements silently disagree with `T`. `S` (the type of
`stats`) is a real parameter, not folded into a bare `stats::NamedTuple`
field: an unparameterised `NamedTuple` field is abstract, so every access
would infer non-concrete and box.

Multiple chains are pooled into one `dim x ndraws` matrix; `nchains` records
how many were combined, and construction refuses an `ndraws` not divisible by
`nchains` (the class of bug in #89: a chain count and a draw count
disagreeing silently). There is no `(iteration, chain)` addressing here — an
engine that needs per-chain diagnostics reaches them through
`trace_to_flexichain` (#91). Indexing `trace[idx]` for a vector `idx` returns
a new trace with `nchains = 1`: slicing can cross chain boundaries, so the
honest result is a single pooled chain rather than a retained count that no
longer describes the data.

# Fields
- `obj`: the template fittable object [`reconstruct`](@ref) rebuilds from
  each draw.
- `draws`: the pooled draws, `flat_dimension(obj) x ndraws`, chains
  concatenated column-wise.
- `names`: the estimated rows' dotted names, in `draws` row order.
- `nchains`: the number of chains pooled into `draws`.
- `stats`: per-draw extras (e.g. log-density, divergence flags); empty until
  an engine populates it (#94).

# See also
- [`draws_to_trace`](@ref): build one from raw sampler draws.
- [`parameter_draws`](@ref), [`trace_to_distribution`](@ref),
  [`point_estimate`](@ref): the transforms built on it.
"
struct PosteriorTrace{T, O, D, S} <: AbstractVector{T}
    obj::O
    draws::Matrix{D}
    names::Vector{Symbol}
    nchains::Int
    stats::S
end

# The one path to a `PosteriorTrace`: computes the phantom `T`, so nothing
# else may call `PosteriorTrace{T,O,D,S}(...)` directly (see the file banner).
function PosteriorTrace(
        obj::O, draws::AbstractMatrix{D}, names::AbstractVector{Symbol},
        nchains::Int; stats::S = NamedTuple()) where {O, D, S}
    nchains >= 1 || throw(ArgumentError(
        "PosteriorTrace: nchains must be at least 1, got $nchains"))
    ndraws = size(draws, 2)
    ndraws > 0 || throw(ArgumentError(
        "PosteriorTrace: got 0 draws; a trace's element type is fixed by " *
        "reconstructing the first draw, which has no answer when there are " *
        "none, so an empty trace cannot be constructed"))
    ndraws % nchains == 0 || throw(ArgumentError(
        "PosteriorTrace: $ndraws draw(s) is not divisible by nchains = " *
        "$nchains; a chain count that disagrees with the draw count would " *
        "pool draws across chains silently"))
    T = typeof(reconstruct(obj, view(draws, :, 1)))
    return PosteriorTrace{T, O, D, S}(
        obj, Matrix{D}(draws), Vector{Symbol}(names), nchains, stats)
end

Base.size(trace::PosteriorTrace) = (size(trace.draws, 2),)

# `reconstruct` is handed a `view`, never a materialised `Vector`: the
# protocol only requires an `AbstractVector`, and an implementation that
# assumes a dense `Vector` (or mutates what it is given) is out of contract —
# see `reconstruct`'s own docstring for the AD-tracer reason its estimated
# fields must stay generically typed regardless of the container.
function Base.getindex(trace::PosteriorTrace, i::Int)
    checkbounds(trace, i)
    return reconstruct(trace.obj, view(trace.draws, :, i))
end

# Slicing returns `nchains = 1` (see the type's own docstring): a subset of
# columns can straddle a chain boundary, so retaining the parent's `nchains`
# would misdescribe the result.
function Base.getindex(trace::PosteriorTrace, idx::AbstractVector{<:Integer})
    checkbounds(trace, idx)
    return PosteriorTrace(trace.obj, trace.draws[:, idx], trace.names, 1;
        stats = _slice_stats(trace.stats, idx))
end

# Generic over `stats`' fields so a per-draw extra populated under #94 slices
# along with the draws it was recorded against; a no-op today since `stats`
# is always empty.
function _slice_stats(stats::NamedTuple, idx)
    return NamedTuple{keys(stats)}(map(v -> v[idx], values(stats)))
end

# `filter` is the one AbstractVector method worth overriding: it is what a
# caller reaches for to drop warmup or divergent draws before calling
# `trace_to_distribution`, and dropping to a plain `Vector` (Julia's default
# for `filter` on an AbstractVector) would silently discard `names`/`nchains`/
# `stats` and make that follow-up call impossible. `map`, `sort`, `vcat` and
# `similar` are left to degrade to a plain `Vector`, by contrast: none of them
# has an obvious element-count-preserving reading against `names`/`nchains`,
# so a `PosteriorTrace`-returning override would have to invent one.
function Base.filter(f, trace::PosteriorTrace)
    return trace[findall(f, trace)]
end

@doc "

Build a [`PosteriorTrace`](@ref) from raw sampler draws.

`draws_to_trace(obj, draws; nchains)` normalises `draws` — a `dim x ndraws`
matrix, or an `ndraws`-length vector of `dim`-length vectors, `dim =
flat_dimension(obj)` — through the same shape check the FlexiChains readback
uses, pools it against `obj`'s estimated rows' dotted names, and returns the
[`PosteriorTrace`](@ref). `nchains` chains are assumed already concatenated
column-wise into `draws`; construction refuses an `ndraws` not divisible by
`nchains`, and refuses `ndraws == 0` (see [`PosteriorTrace`](@ref) for both).
An object estimating nothing (`dim == 0`) still needs `draws` to carry the
draw count: pass a `(0, ndraws)` matrix or an `ndraws`-length vector of empty
vectors.

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `draws`: the raw draws, `dim x ndraws` or an `ndraws`-vector of
  `dim`-vectors.

# Keyword Arguments
- `nchains`: the number of chains pooled column-wise into `draws` (default
  `1`).

# Examples
```@example
using DistributionsInference, Distributions

struct TraceLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::TraceLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(d::TraceLeaf, x::AbstractVector)
    return TraceLeaf(x[1], d.scale)
end

leaf = TraceLeaf(2.0, 1.0)
trace = draws_to_trace(leaf, reshape([2.1, 2.4, 2.0, 2.6], 1, :))
length(trace)
```

# See also
- [`PosteriorTrace`](@ref): the type this builds.
- [`parameter_draws`](@ref), [`trace_to_distribution`](@ref),
  [`point_estimate`](@ref): the transforms built on it.
"
function draws_to_trace(obj, draws::AbstractMatrix; nchains::Int = 1)
    mat = _draws_matrix(draws, flat_dimension(obj))
    return _draws_to_trace(obj, mat, nchains)
end

function draws_to_trace(
        obj, draws::AbstractVector{<:AbstractVector}; nchains::Int = 1)
    mat = _draws_matrix(draws, flat_dimension(obj))
    return _draws_to_trace(obj, mat, nchains)
end

# Neither accepted raw shape's method matched, same fallback `_to_flexichain`
# uses for the same reason.
function draws_to_trace(obj, draws; nchains::Int = 1)
    return _malformed_draws(draws)
end

function _draws_to_trace(obj, mat::AbstractMatrix, nchains::Int)
    names = Symbol[row.name for row in estimated_rows(obj)]
    return PosteriorTrace(obj, mat, names, nchains)
end

# `distribution_params`'s duplicate-name guard (`readback.jl`, via the
# FlexiChains extension) re-derived here without the `FlexiChains` tie: a
# repeated dotted name is a `parameter_rows` protocol bug independent of
# which readback path (chain or trace) exposed it.
function _check_unique_trace_names(names::AbstractVector{Symbol})
    length(names) == length(Set(names)) && return nothing
    counts = Dict{Symbol, Int}()
    for n in names
        counts[n] = get(counts, n, 0) + 1
    end
    dupes = [n for (n, c) in counts if c > 1]
    throw(ArgumentError(
        "parameter_draws: duplicate estimated parameter name(s) $(dupes); " *
        "parameter_rows(obj) must give every estimated row a unique dotted " *
        "name"))
end

@doc "

Per-parameter posterior draws from a trace, unreduced.

`parameter_draws(trace)` returns a `NamedTuple` keyed by `trace.names`, each
entry the full row of draws for that estimated parameter (a view into
`trace.draws`, not a reduction). Where [`point_estimate`](@ref) collapses a
trace to one reconstructed object, `parameter_draws` keeps every draw as one
vector per parameter — the trace-side counterpart of
[`distribution_draws`](@ref), which instead keeps one reconstructed object
per draw.

# Arguments
- `trace`: the [`PosteriorTrace`](@ref) to read.

# Examples
```@example
using DistributionsInference, Distributions

struct ParamsTraceLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::ParamsTraceLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(
        d::ParamsTraceLeaf, x::AbstractVector)
    return ParamsTraceLeaf(x[1], d.scale)
end

leaf = ParamsTraceLeaf(2.0, 1.0)
trace = draws_to_trace(leaf, reshape([2.1, 2.4, 2.0, 2.6], 1, :))
parameter_draws(trace).shape
```

# See also
- [`point_estimate`](@ref): the reduced, single-object form.
- [`distribution_draws`](@ref): the chain-side equivalent (one object per
  draw, rather than one vector per parameter).
"
function parameter_draws(trace::PosteriorTrace)
    names = trace.names
    _check_unique_trace_names(names)
    return NamedTuple{Tuple(names)}(
        Tuple(view(trace.draws, i, :) for i in eachindex(names)))
end

@doc "

The Monte Carlo posterior predictive, as a mixture over a trace's draws.

`trace_to_distribution(trace)` returns an equally-weighted
`Distributions.MixtureModel` over `collect(trace)`: `trace.obj` reconstructed
at every draw, mixed together, so the returned distribution's `pdf`, `rand`,
`mean`, ... integrate over the posterior uncertainty the trace carries. Every
draw is weighted equally, matching an unweighted Monte Carlo trace — this is
the *mean distribution* that [`Statistics.mean`](@ref)`(::PosteriorTrace)`'s
error message points at, as distinct from [`point_estimate`](@ref)'s
*distribution at the mean parameters*.

Needs `eltype(trace) <: Distributions.Distribution`: a trace of a plain
struct (e.g. this package's own toy fixtures, which carry a `logpdf` method
rather than subtyping `Distribution`) has nothing for `MixtureModel` to mix,
and is refused with an `ArgumentError` naming the actual element type rather
than failing inside `MixtureModel`'s own constructor.

# Arguments
- `trace`: the [`PosteriorTrace`](@ref) to mix over; `eltype(trace)` must be
  a `Distributions.Distribution`.

# Examples
```@example
using DistributionsInference, Distributions

struct DistTraceLeaf
    shape::Float64
end

function DistributionsInference.parameter_rows(d::DistTraceLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(d::DistTraceLeaf, x::AbstractVector)
    return Gamma(x[1], 1.0)
end

leaf = DistTraceLeaf(2.0)
trace = draws_to_trace(leaf, reshape([2.1, 2.4, 2.0, 2.6], 1, :))
mixture = trace_to_distribution(trace)
mean(mixture)
```

# See also
- [`point_estimate`](@ref): the distribution at the mean/summary parameters,
  not the mean distribution.
- [`Statistics.mean`](@ref): refuses on a `PosteriorTrace`, naming both of
  the above.
"
function trace_to_distribution(trace::PosteriorTrace{T}) where {T}
    T <: Distributions.Distribution || throw(ArgumentError(
        "trace_to_distribution(trace) needs eltype(trace) <: " *
        "Distributions.Distribution to mix over; got $T. reconstruct(obj, " *
        "...) returns a $T here, not a `Distribution` — read this trace " *
        "back with parameter_draws/point_estimate instead."))
    return Distributions.MixtureModel(collect(trace))
end

@doc "

The plug-in distribution at a trace's summarised parameters.

`point_estimate(trace; summary = mean)` reduces every estimated parameter's
draws with `summary` (default the posterior mean) and calls
[`reconstruct`](@ref)`(trace.obj, ...)` on the result: one object, at one set
of parameter values, as opposed to [`trace_to_distribution`](@ref)'s mixture
over every draw. `summary` stays a keyword default rather than a required
positional argument: the reduction has a legitimate use handing one
`Distribution` to code that takes one, and the name already carries the
warning the plain readback did not.

# Arguments
- `trace`: the [`PosteriorTrace`](@ref) to summarise.

# Keyword Arguments
- `summary`: the reduction `AbstractVector -> scalar` applied to each
  parameter's draws (default `Statistics.mean`).

# Examples
```@example
using DistributionsInference, Distributions
using Statistics: median

struct PointTraceLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::PointTraceLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(
        d::PointTraceLeaf, x::AbstractVector)
    return PointTraceLeaf(x[1], d.scale)
end

leaf = PointTraceLeaf(2.0, 1.0)
trace = draws_to_trace(leaf, reshape([2.1, 2.4, 2.0, 2.6], 1, :))
point_estimate(trace; summary = median).shape
```

# See also
- [`trace_to_distribution`](@ref): the mixture-over-every-draw alternative.
- [`parameter_draws`](@ref): the unreduced draws this summarises.
"
function point_estimate(trace::PosteriorTrace; summary = Statistics.mean)
    names = trace.names
    isempty(names) && return reconstruct(trace.obj, Float64[])
    x = [summary(view(trace.draws, i, :)) for i in eachindex(names)]
    return reconstruct(trace.obj, x)
end

# Refuse a reduction directly on a `PosteriorTrace`, consistently across
# every reducer this ambiguity applies to (see the `@doc` block below for the
# reasoning; this is shared so `mean`/`sum`/`median`/`std`/`var`/`quantile`
# give the identical message rather than six near-duplicates that drift).
# Without this, each one fails anyway (e.g. `+` or `isless` has no method on
# the trace's element type), just two frames deeper inside `Statistics`/
# `Base`, naming that element type rather than the trace.
function _reduction_ambiguous(f::Symbol)
    return ArgumentError(
        "$(f)(trace) is ambiguous for a PosteriorTrace: it could mean " *
        "$f applied to trace_to_distribution(trace) (the posterior-" *
        "predictive mixture over every draw), or point_estimate(trace; " *
        "summary = $f) (the distribution at the $f-summarised parameters). " *
        "Call one of those directly; $f.(trace) still broadcasts $f over " *
        "each reconstructed element, unaffected by this.")
end

@doc "

Refuse the ambiguous `mean` of a trace.

`Statistics.mean(::PosteriorTrace)` has no method: \"the mean of a trace\" is
ambiguous between the two things this redesign keeps apart — the *mean
distribution*, [`trace_to_distribution`](@ref)`(trace)`'s posterior-
predictive mixture, and the *distribution at the mean parameters*,
[`point_estimate`](@ref)`(trace)` (`summary = mean` by default). `sum`,
`Statistics.median`, `Statistics.std`, `Statistics.var` and non-broadcast
`Statistics.quantile(trace, p)` carry the same ambiguity (`summary = median`,
`summary = std`, ... in place of `mean`) and are refused the same way, all
through one private helper so the six messages cannot drift apart. These
methods exist only to name the two alternatives, rather than let the reducer
resolve through
`AbstractVector`'s generic fallback (`+`/`isless` on `T`, usually) to a
confusing failure two frames deeper that never mentions the trace.

Broadcasting is unaffected: `mean.(trace)` (and `quantile.(trace, p)`, ...)
map the reducer over each *element* of `trace` (each
[`reconstruct`](@ref)ed object in turn), which never calls these methods —
only the un-broadcast call on the trace itself does.

# Arguments
- `trace`: the [`PosteriorTrace`](@ref) the reducer was called on.

# See also
- [`trace_to_distribution`](@ref), [`point_estimate`](@ref): the two things
  every one of these points at.
"
function Statistics.mean(trace::PosteriorTrace)
    throw(_reduction_ambiguous(:mean))
end

function Base.sum(trace::PosteriorTrace)
    throw(_reduction_ambiguous(:sum))
end

function Statistics.median(trace::PosteriorTrace)
    throw(_reduction_ambiguous(:median))
end

function Statistics.std(trace::PosteriorTrace)
    throw(_reduction_ambiguous(:std))
end

function Statistics.var(trace::PosteriorTrace)
    throw(_reduction_ambiguous(:var))
end

function Statistics.quantile(trace::PosteriorTrace, p::Real)
    throw(_reduction_ambiguous(:quantile))
end
