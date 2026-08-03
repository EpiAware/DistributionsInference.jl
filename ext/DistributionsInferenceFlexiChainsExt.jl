# DistributionsInference x FlexiChains: the dotted-name chain readback — build
# a `FlexiChain` from the raw draws any `LogDensityProblems`-compatible sampler
# hands back (`to_flexichain`), read its estimated values keyed by dotted name
# (`distribution_params`), and read the chain back onto a fitted object
# (`readback`, `readback_draws`). The four functions are declared with their
# docstrings in `src/readback.jl`; only their methods live here, loaded when
# `FlexiChains` is available, so the core package carries no chain dependency
# and an object is fittable (and its log-density sampleable) without one.
#
# No PPL is involved anywhere in this file: the naming contract this
# generalises (ComposedDistributions' `chain_to_params`/`param_draws`,
# ComposedDistributions#185) is over the fit protocol's own dotted row names.
# A `VarName`-keyed chain sampled from `as_turing` is renamed onto those same
# dotted names by `DistributionsInferenceDynamicPPLFlexiChainsExt` and handed
# straight to the methods here.
module DistributionsInferenceFlexiChainsExt

using DistributionsInference: DistributionsInference, estimated_rows,
                              reconstruct, _draws_matrix
import DistributionsInference: to_flexichain, distribution_params, readback,
                               readback_draws
using FlexiChains: FlexiChains
using Statistics: mean

# `to_flexichain` is typed on each raw draw shape rather than declared once
# with the stub's own `(obj, draws)` signature: an identical signature here
# would OVERWRITE the core method (the one that names this extension when it
# is missing) instead of adding to it. The core stub keeps the "neither shape"
# rejection for anything these two do not match.
to_flexichain(obj, draws::AbstractMatrix) = _to_flexichain(obj, draws)
to_flexichain(obj, draws::AbstractVector) = _to_flexichain(obj, draws)

function _to_flexichain(obj, draws)
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
    draws isa Function &&
        return [i for i in 1:FlexiChains.niters(chain) if draws(i)]
    return collect(draws)
end

_select_draws(col, ::Colon) = vec(col)
_select_draws(col, sel) = vec(col)[sel]

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

function readback(obj, chain::FlexiChains.FlexiChain; summary = mean,
        draw = nothing, draws = nothing)
    nt = distribution_params(obj, chain; summary = summary, draw = draw,
        draws = draws)
    return reconstruct(obj, collect(values(nt)))
end

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

end # module DistributionsInferenceFlexiChainsExt
