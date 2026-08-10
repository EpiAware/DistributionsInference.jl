# The posterior-output API: three ways to turn a fitted chain (or raw draws)
# back into a `Distribution`, replacing `point_estimate`/`distribution_draws`
# for a caller who wants a `Distribution` out rather than a rebuilt fittable
# object. Additive: `point_estimate`, `distribution_params` and
# `distribution_draws` are unaffected and keep working.
#
# Each of the three has ONE concrete return type, which is why this is
# dispatch (three names) rather than one function branching on a keyword:
#
#   inference_to_distribution(obj, chain; draws)        -> MixtureModel
#   inference_to_distributions(obj, chain; draws)        -> Vector{<:Distribution}
#   inference_to_distribution(obj, chain, summary; draws) -> one Distribution
#
# `chain` also accepts a raw `dim x ndraws` matrix or a vector of `dim`-length
# vectors (with an `nchains` keyword), built into a `FlexiChain` via
# `_to_flexichain` and dispatched on again.

# An explicit range/vector selection must fall inside the pooled range: the
# whole point of documenting the chain-major trap is that a caller ends up
# hand-writing indices after reading it, and that is exactly the caller who
# needs a directed error rather than a raw `BoundsError` two frames into
# column indexing. `extrema` is a single O(length) pass (O(1) for a range),
# and reports the actual out-of-range value reached, not just "somewhere
# out of bounds". Emptiness is not a bounds problem and is handled
# separately (by the callers that require a nonempty selection).
function _check_pooled_bounds(n::Int, idx)
    isempty(idx) && return nothing
    lo, hi = extrema(idx)
    lo >= 1 || throw(ArgumentError(
        "draws index $lo is out of range: pooled draw indices run 1:$n " *
        "(indices below 1, including 0 and negative indices, are not " *
        "valid — there is no wraparound-from-the-end indexing here)"))
    hi <= n || throw(ArgumentError(
        "draws index $hi is out of range: the pooled draw count is $n, " *
        "so valid indices run 1:$n"))
    return nothing
end

# The `draws` keyword: `nothing` (every draw), an `AbstractRange` or
# `AbstractVector{<:Integer}` of exact pooled indices, or an `Integer` number
# of draws sampled at random from the pooled set (without replacement).
# Deliberately no predicate form: `findall` expresses that over the pooled
# range directly, and the old predicate branch in `_draw_indices` is exactly
# where #89's pooled-range bug lived.
_resolve_pooled_draws(n::Int, ::Nothing, ::Any) = 1:n

function _resolve_pooled_draws(n::Int, draws::AbstractRange{<:Integer}, ::Any)
    _check_pooled_bounds(n, draws)
    return draws
end

function _resolve_pooled_draws(
        n::Int, draws::AbstractVector{<:Integer}, ::Any)
    _check_pooled_bounds(n, draws)
    return draws
end

function _resolve_pooled_draws(n::Int, k::Integer, rng)
    0 <= k <= n || throw(ArgumentError(
        "draws=$k requests more draws than the $n pooled across every " *
        "chain; pass an Integer no greater than the pooled draw count, or " *
        "an explicit index selection"))
    # `randperm` allocates a length-n permutation regardless of k, rather
    # than a partial without-replacement sample. n is the pooled draw
    # count (thousands at most in practice) and reconstruct(obj, x) below
    # dominates the cost by 1-2 orders of magnitude even at k << n, so the
    # simpler, well-tested stdlib primitive wins over a hand-rolled partial
    # shuffle here.
    return sort!(Random.randperm(rng, n)[1:k])
end

function _resolve_pooled_draws(n::Int, draws, ::Any)
    throw(ArgumentError(
        "draws must be `nothing` (every draw), an `AbstractRange` or " *
        "`AbstractVector{<:Integer}` of pooled draw indices, or an " *
        "`Integer` number of draws to sample at random; got " *
        "$(typeof(draws))"))
end

# `inference_to_distribution` (both the mixture and the plug-in forms) needs
# at least one draw: a `MixtureModel` over zero components fails inside
# Distributions.jl naming neither this function nor the cause, and a
# `summary` over an empty collection either errors unhelpfully or (`mean`)
# silently returns `NaN`. `k = 0` and `draws = <empty vector>` both resolve
# to an empty selection, so this is checked on the resolved selection, not
# on the `draws` argument's own emptiness. `inference_to_distributions`
# deliberately has no such guard: an empty selection there is a legitimate
# empty `Vector` result.
function _require_nonempty_selection(f::Symbol, idx)
    isempty(idx) && throw(ArgumentError(
        "`$f` needs at least one selected draw; the resolved `draws` " *
        "selection is empty. Use `inference_to_distributions` if an empty " *
        "result is what you want."))
    return idx
end

# Reduce each estimated row's selected draws by `summary`, then `reconstruct`
# once: the plug-in used by the 3-argument `inference_to_distribution`.
function _reconstruct_summary(obj, chain::FlexiChains.FlexiChain, summary, idx)
    rows = estimated_rows(obj)
    isempty(rows) && return reconstruct(obj, Float64[])
    vals = [summary(vec(_chain_column(chain, row.name))[idx]) for row in rows]
    return reconstruct(obj, vals)
end

# `inference_to_distribution`/`inference_to_distribution(..., summary)` need
# `reconstruct(obj, x)` to return a `Distribution`; `inference_to_distributions`
# does not (it is the generic vectorised readback, e.g. for a non-`Distribution`
# fittable object). This is the shared check, naming the actual type reached.
function _require_distribution(f::Symbol, x)
    x isa Distributions.Distribution && return x
    throw(ArgumentError(
        "`$f` needs `reconstruct(obj, x)` to return a `Distribution`; got " *
        "$(typeof(x)). Use `inference_to_distributions` instead when the " *
        "reconstructed object is not a `Distribution`."))
end

function _require_distribution_eltype(f::Symbol, dists::AbstractVector)
    eltype(dists) <: Distributions.Distribution && return dists
    throw(ArgumentError(
        "`$f` needs `reconstruct(obj, x)` to return a `Distribution`; got " *
        "element type $(eltype(dists)). Use `inference_to_distributions` " *
        "instead when the reconstructed object is not a `Distribution`."))
end

@doc "

Every selected draw, reconstructed: the vectorised posterior readback.

`inference_to_distributions(obj, chain; draws = nothing)` reads `chain`
(pooled across every chain it carries) and returns one
[`reconstruct`](@ref)`(obj, x)` per selected draw. Unlike
[`inference_to_distribution`](@ref), the reconstructed element does not need
to be a `Distribution` — this is the generic vectorised readback, and the
additive replacement for [`distribution_draws`](@ref) (identical selected
values, a different `draws` selector).

`chain` is a dotted-name `FlexiChain`, a `VarName`-keyed chain (once
`DynamicPPL` is loaded alongside this package), or raw draws with no chain of
their own — a `dim x ndraws` matrix or an `ndraws`-length vector of
`dim`-length vectors, built into a chain via `_to_flexichain` (internal) — with
an `nchains` keyword (default `1`) for the raw-draws form.

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `chain`: the chain (or raw draws) to read from.

# Keyword Arguments
- `draws`: which pooled draws to keep — see \"The `draws` keyword\" below.
- `nchains`: raw-draws form only; splits the pooled columns into this many
  equal chains, chain-major (default `1`).
- `rng`: the random-number generator used when `draws` is an `Integer`
  (default `Random.default_rng()`).

# The `draws` keyword

Every chain this reads pools its iterations across every chain it carries,
**chain-major**: chain 1 occupies the first `niters` pooled entries, chain 2
occupies the next `niters`, and so on. `draws = 1:200` on a chain with 4
chains of 2000 iterations each returns 200 draws from chain 1 only — it looks
like a subsample of the whole run and is not (the user-facing form of the
bug fixed in #89).

- `nothing` (default): every pooled draw.
- an `AbstractRange` or `AbstractVector{<:Integer}`: exactly those pooled
  indices, in the chain-major order above. Every index must fall within
  `1:n` (`n` the pooled draw count) — an out-of-range index (including `0`
  or negative: there is no wraparound-from-the-end indexing here) raises an
  `ArgumentError` naming `n` and the offending index, rather than reaching a
  bare `BoundsError`.
- an `Integer` `n`: `n` draws sampled at random *from the whole pooled set*,
  so this — not a range — is the way to get \"some draws\" that actually span
  every chain. `0 <= n <= (the pooled draw count)`, checked the same way.

An empty selection (`draws = 0`, or an empty range/vector) is allowed here
and returns an empty `Vector` — there is nothing wrong with \"zero draws,
reconstructed\" as a vectorised result.
[`inference_to_distribution`](@ref) (both forms) refuses an empty selection
instead, since a `MixtureModel` over zero components and a `summary` over an
empty collection have no sensible answer.

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
length(inference_to_distributions(leaf, chain))
```

# See also
- [`inference_to_distribution`](@ref): the equal-weight `MixtureModel` over
  this same selection — the Monte Carlo posterior predictive.
- [`inference_to_distribution`](@ref)`(obj, chain, summary)`: the plug-in
  point estimate instead of the full draw set.
- [`distribution_draws`](@ref): the function this replaces.
"
function inference_to_distributions(obj, chain::FlexiChains.FlexiChain;
        draws = nothing, rng = Random.default_rng())
    idx = _resolve_pooled_draws(_pooled_ndraws(chain), draws, rng)
    return _reconstruct_pooled(obj, chain, idx)
end

function inference_to_distributions(obj, chain; kwargs...)
    return _wrong_chain_type(:inference_to_distributions, chain)
end

function inference_to_distributions(
        obj, raw::Union{AbstractMatrix, AbstractVector{<:AbstractVector}};
        draws = nothing, nchains::Int = 1, rng = Random.default_rng())
    return inference_to_distributions(
        obj, _to_flexichain(obj, raw; nchains = nchains);
        draws = draws, rng = rng)
end

@doc "

The equal-weight `MixtureModel` over selected draws: the Monte Carlo
posterior predictive.

`inference_to_distribution(obj, chain; draws = nothing)` reconstructs one
`Distribution` per selected draw (as [`inference_to_distributions`](@ref)
does) and returns the equal-weight `MixtureModel` over them — the Monte Carlo
approximation to the posterior predictive distribution, carrying the full
posterior uncertainty rather than collapsing it to a point.

Raises an `ArgumentError` naming the actual type reached when
[`reconstruct`](@ref)`(obj, x)` does not return a `Distribution` (e.g. the
toy fittable types in this package's own docstrings) — use
[`inference_to_distributions`](@ref) for those.

`chain` also accepts raw draws with no chain of their own; see
[`inference_to_distributions`](@ref) for the accepted forms and the `nchains`
keyword.

# Performance

The `MixtureModel` is built over *every* selected draw — this function does
not cap or thin it, even at a large draw count, so a call over the pooled
draws of a big multi-chain run mixes every one of them. Selecting fewer
draws (via the `draws` keyword) is the caller's tool for the cost below, not
a hidden default.

Once built, `mean`/`rand`/`pdf`/`cdf` on the mixture are cheap: `mean`/`rand`
are `O(1)` in the component count `K`, and `pdf`/`cdf` are `O(K)` per call but
stay well under a millisecond even at `K = 8000` (~0.16 ms / ~0.9 ms
respectively). `quantile` is the one call that is not cheap: it root-finds
against the mixture's `O(K)` `cdf` on every iteration, so its cost grows with
`K` and reaches roughly 13 ms at `K = 8000`. A single `quantile` call is
fine; many `quantile` calls in a loop (e.g. a full posterior interval grid)
are where this adds up — thin first with `draws = 500` (or similar) to cut
`K`, and `quantile`'s cost, by the same factor.

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `chain`: the chain (or raw draws) to read from.

# Keyword Arguments
- `draws`: which pooled draws to keep. See
  [`inference_to_distributions`](@ref) for the four accepted forms and the
  chain-major pooling this documents (`draws = 1:n` is *not* a subsample of a
  multi-chain run — read that docstring before using a range here). Also the
  cheapest way to bound the `quantile` cost above, by capping `K`.
- `nchains`: raw-draws form only (default `1`).
- `rng`: used when `draws` is an `Integer` (default `Random.default_rng()`).

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChain, Parameter

struct GammaLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::GammaLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(d::GammaLeaf, x::AbstractVector)
    return Gamma(x[1], d.scale)
end

leaf = GammaLeaf(2.0, 1.0)
draws = [2.1, 2.4, 2.0, 2.6]
chain = FlexiChain{Symbol}(
    4, 1, Dict(Parameter(:shape) => reshape(draws, 4, 1)))
mean(inference_to_distribution(leaf, chain))
```

# See also
- [`inference_to_dist`](@ref): an alias for this function.
- [`inference_to_distributions`](@ref): the vectorised form this is built on,
  and the `draws` keyword's full documentation.
- [`inference_to_distribution`](@ref)`(obj, chain, summary)`: the cheaper
  plug-in point estimate this trades accuracy against.
"
function inference_to_distribution(obj, chain::FlexiChains.FlexiChain;
        draws = nothing, rng = Random.default_rng())
    dists = inference_to_distributions(obj, chain; draws = draws, rng = rng)
    _require_nonempty_selection(:inference_to_distribution, dists)
    _require_distribution_eltype(:inference_to_distribution, dists)
    return Distributions.MixtureModel(dists)
end

function inference_to_distribution(obj, chain; kwargs...)
    return _wrong_chain_type(:inference_to_distribution, chain)
end

function inference_to_distribution(
        obj, raw::Union{AbstractMatrix, AbstractVector{<:AbstractVector}};
        draws = nothing, nchains::Int = 1, rng = Random.default_rng())
    return inference_to_distribution(
        obj, _to_flexichain(obj, raw; nchains = nchains);
        draws = draws, rng = rng)
end

@doc "

The marginal plug-in `Distribution`: summarise, then reconstruct once.

`inference_to_distribution(obj, chain, summary; draws = nothing)` reduces
each estimated row's selected draws with `summary` (e.g. `mean`) to a flat
parameter vector, and calls [`reconstruct`](@ref) *once* on that vector — the
plug-in estimate, positionally requiring the reduction so the loss of
posterior uncertainty is visible at the call site. This replaces the old
`point_estimate`.

**This is not the posterior predictive.** Summarising before reconstructing
is not the same distribution as reconstructing every draw and mixing:
`Gamma(mean(shape draws), scale) != mean(Gamma.(shape draws, scale))` — the
left side is what this function returns, the right side (its Monte Carlo
approximation) is [`inference_to_distribution`](@ref)`(obj, chain)`. Reach
for this only when a single point estimate genuinely suffices (e.g. handing
one `Distribution` to code that takes one), not as a stand-in for the
posterior predictive.

Raises an `ArgumentError` naming the actual type reached when
[`reconstruct`](@ref)`(obj, x)` does not return a `Distribution`.

`chain` also accepts raw draws with no chain of their own; see
[`inference_to_distributions`](@ref) for the accepted forms and the
`nchains` keyword.

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `chain`: the chain (or raw draws) to read from.
- `summary`: the reduction `AbstractVector -> scalar` applied to each
  estimated row's selected draws (e.g. `mean`, `median`).

# Keyword Arguments
- `draws`: which pooled draws to summarise over. See
  [`inference_to_distributions`](@ref) for the four accepted forms and the
  chain-major pooling caveat.
- `nchains`: raw-draws form only (default `1`).
- `rng`: used when `draws` is an `Integer` (default `Random.default_rng()`).

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChain, Parameter
using Statistics: mean

struct GammaLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::GammaLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(d::GammaLeaf, x::AbstractVector)
    return Gamma(x[1], d.scale)
end

leaf = GammaLeaf(2.0, 1.0)
draws = [2.1, 2.4, 2.0, 2.6]
chain = FlexiChain{Symbol}(
    4, 1, Dict(Parameter(:shape) => reshape(draws, 4, 1)))
inference_to_distribution(leaf, chain, mean)
```

# See also
- [`inference_to_distribution`](@ref): the Monte Carlo posterior predictive
  this trades accuracy for cheapness against.
- [`point_estimate`](@ref): the function this replaces.
"
function inference_to_distribution(
        obj, chain::FlexiChains.FlexiChain, summary;
        draws = nothing, rng = Random.default_rng())
    idx = _resolve_pooled_draws(_pooled_ndraws(chain), draws, rng)
    _require_nonempty_selection(:inference_to_distribution, idx)
    result = _reconstruct_summary(obj, chain, summary, idx)
    return _require_distribution(:inference_to_distribution, result)
end

function inference_to_distribution(obj, chain, summary; kwargs...)
    return _wrong_chain_type(:inference_to_distribution, chain)
end

function inference_to_distribution(
        obj, raw::Union{AbstractMatrix, AbstractVector{<:AbstractVector}},
        summary; draws = nothing, nchains::Int = 1, rng = Random.default_rng())
    return inference_to_distribution(
        obj, _to_flexichain(obj, raw; nchains = nchains), summary;
        draws = draws, rng = rng)
end

@doc "

Short alias for [`inference_to_distribution`](@ref).

`inference_to_dist(obj, chain; draws)` is identical to
`inference_to_distribution(obj, chain; draws)` — the equal-weight
`MixtureModel` over selected draws — and `inference_to_dist(obj, chain,
summary; draws)` is identical to the marginal plug-in form (reduction
positional). `inference_to_distribution` is the canonical, documented name;
this is a shorter spelling of the *same underlying function* (a `const`
alias, so every method — including one an extension adds later — is
automatically shared between the two names).

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `chain`: the chain (or raw draws) to read from.
- `summary`: (3-argument form only) the reduction applied to each estimated
  row's selected draws before a single [`reconstruct`](@ref).

# Keyword Arguments
- `draws`: which pooled draws to use; see
  [`inference_to_distributions`](@ref) for the accepted forms and the
  chain-major pooling caveat.
- `nchains`: raw-draws form only (default `1`).
- `rng`: used when `draws` is an `Integer` (default `Random.default_rng()`).

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChain, Parameter

struct GammaLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::GammaLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(d::GammaLeaf, x::AbstractVector)
    return Gamma(x[1], d.scale)
end

leaf = GammaLeaf(2.0, 1.0)
draws = [2.1, 2.4, 2.0, 2.6]
chain = FlexiChain{Symbol}(
    4, 1, Dict(Parameter(:shape) => reshape(draws, 4, 1)))
mean(inference_to_dist(leaf, chain))
```

# See also
- [`inference_to_distribution`](@ref): the canonical name this aliases.
"
const inference_to_dist = inference_to_distribution

@doc "

Short alias for [`inference_to_distributions`](@ref).

`inference_to_dists(obj, chain; draws)` is identical to
`inference_to_distributions(obj, chain; draws)` — every selected draw,
reconstructed. `inference_to_distributions` is the canonical, documented
name; this is a shorter spelling of the *same underlying function* (a
`const` alias, so every method — including one an extension adds later — is
automatically shared between the two names).

# Arguments
- `obj`: the fittable object the draws were sampled for.
- `chain`: the chain (or raw draws) to read from.

# Keyword Arguments
- `draws`: which pooled draws to keep; see
  [`inference_to_distributions`](@ref) for the accepted forms and the
  chain-major pooling caveat.
- `nchains`: raw-draws form only (default `1`).
- `rng`: used when `draws` is an `Integer` (default `Random.default_rng()`).

# Examples
```@example
using DistributionsInference, Distributions
using FlexiChains: FlexiChain, Parameter

struct DrawsAliasLeaf
    shape::Float64
    scale::Float64
end

function DistributionsInference.parameter_rows(d::DrawsAliasLeaf)
    return [(name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end
function DistributionsInference.reconstruct(
        d::DrawsAliasLeaf, x::AbstractVector)
    return DrawsAliasLeaf(x[1], d.scale)
end

leaf = DrawsAliasLeaf(2.0, 1.0)
draws = [2.1, 2.4, 2.0, 2.6]
chain = FlexiChain{Symbol}(
    4, 1, Dict(Parameter(:shape) => reshape(draws, 4, 1)))
length(inference_to_dists(leaf, chain))
```

# See also
- [`inference_to_distributions`](@ref): the canonical name this aliases.
"
const inference_to_dists = inference_to_distributions
