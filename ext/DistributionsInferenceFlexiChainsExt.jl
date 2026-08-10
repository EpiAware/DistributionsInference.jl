# DistributionsInference x FlexiChains: the dotted-name chain readback. Read a
# chain's estimated values by dotted name (`distribution_params`), and read a
# chain back onto a fitted object (`point_estimate`, `distribution_draws`), plus
# the internal raw-draws constructor (`_to_flexichain`). All four are declared
# with their docstrings in `src/readback.jl`.
module DistributionsInferenceFlexiChainsExt

using DistributionsInference: DistributionsInference, estimated_rows,
                              reconstruct, _draws_matrix
import DistributionsInference: _to_flexichain, distribution_params,
                               point_estimate, distribution_draws
using FlexiChains: FlexiChains
using Statistics: mean

# Typed on each raw draw shape rather than the stub's own `(obj, draws)`
# signature: an identical signature would overwrite the core method (the one
# that names this extension when it is missing) instead of adding to it.
_to_flexichain(obj, draws::AbstractMatrix) = _chain_from_draws(obj, draws)
_to_flexichain(obj, draws::AbstractVector) = _chain_from_draws(obj, draws)

function _chain_from_draws(obj, draws)
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

function _chain_column(chain, name::Symbol)
    FlexiChains.has_parameter(chain, name) ||
        throw(ArgumentError("parameter $(repr(name)) not found in chain"))
    return chain[name]
end

# The number of draws pooled across every chain: `_chain_column` returns the
# full niters x nchains array and `vec` flattens it column-major, so the
# pooled range a draw selector or an index into it must span is
# niters * nchains, not niters alone (one chain's worth).
_pooled_ndraws(chain) = FlexiChains.niters(chain) * FlexiChains.nchains(chain)

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

function point_estimate(obj, chain::FlexiChains.FlexiChain; summary = mean,
        draw = nothing, draws = nothing)
    nt = distribution_params(obj, chain; summary = summary, draw = draw,
        draws = draws)
    return reconstruct(obj, collect(values(nt)))
end

function distribution_draws(obj, chain::FlexiChains.FlexiChain; draws = nothing)
    rows = estimated_rows(obj)
    sel = _draw_indices(chain, draws)
    idx = sel isa Colon ? (1:_pooled_ndraws(chain)) : sel
    isempty(rows) && return [reconstruct(obj, Float64[]) for _ in idx]
    # Materialise each column once so this stays O(niter) rather than
    # re-extracting every column per draw.
    cols = [vec(_chain_column(chain, row.name)) for row in rows]
    return [reconstruct(obj, [col[i] for col in cols]) for i in idx]
end

end # module DistributionsInferenceFlexiChainsExt
