module DistributionsInferenceDynamicPPLFlexiChainsExt

# DistributionsInference x DynamicPPL x FlexiChains: the `VarName`-keyed chain
# readback (DI#4). A chain sampled from `as_turing` is keyed by `VarName` (e.g.
# via `chain_type = FlexiChains.VNChain`), so it is renamed onto the estimated
# rows' dotted `Symbol` names and handed to the `Symbol`-keyed readback in
# `DistributionsInferenceFlexiChainsExt`. The docstrings on
# `distribution_params`/`readback`/`readback_draws` (`src/readback.jl`) stay
# the single source of truth for all of them.
#
# This is a genuine intersection of the two packages, hence its own extension
# rather than a share of either single-package one: the `VarName` naming comes
# from DynamicPPL and the chain from FlexiChains, while `as_turing`
# (`DistributionsInferenceDynamicPPLExt`) needs no chain type at all and the
# `Symbol`-keyed readback (`DistributionsInferenceFlexiChainsExt`) needs no
# PPL. Loading `DynamicPPL` alone still gets the model; loading `FlexiChains`
# alone still gets the readback.

using DistributionsInference: DistributionsInference, estimated_rows,
                              _row_varname
import DistributionsInference: distribution_params, readback, readback_draws
using DynamicPPL: VarName
using FlexiChains: FlexiChains

# Rename a `VarName`-keyed chain's parameters onto the estimated rows' dotted
# `Symbol` names, so it matches what `to_flexichain` would have built and the
# `Symbol`-keyed readback machinery (`distribution_params`/`_chain_column` in
# `ext/DistributionsInferenceFlexiChainsExt.jl`) reads it unchanged. A chain
# parameter that is not one of `obj`'s estimated rows' `VarName`s signals a
# chain that was not sampled from `as_turing(obj, ...)` at this `prefix`
# (wrong prefix, or a mismatched template), so it errors rather than silently
# dropping the column. `_row_varname` is the DynamicPPL extension's own naming
# method, reached here by ordinary dispatch on the core stub, so the two
# extensions cannot drift apart on how a row name becomes a `VarName`.
function _to_symbol_chain(obj, chain::FlexiChains.FlexiChain{<:VarName},
        prefix::Symbol)
    lookup = Dict(_row_varname(prefix, row.name) => row.name
    for row in estimated_rows(obj))
    if isempty(FlexiChains.parameters(chain))
        # `FlexiChains.map_parameters` infers the wrong key type on a CHAIN
        # with zero parameters (an empty `Set` there resolves to `Union{}`,
        # not `Symbol`, and building a `FlexiChain{Union{}}` stack-overflows
        # on its own `NamedTuple` reconstruction): build a fresh empty
        # `Symbol`-keyed chain directly instead, mirroring `to_flexichain`'s
        # own 0-estimated construction. Guarding on the CHAIN being empty
        # (not `lookup`/`obj`'s estimated rows) matters: an `obj` with
        # estimated rows against a genuinely empty chain is a mismatch, and
        # still needs to reach the readback's `has_parameter` check to be
        # reported as one, rather than silently reading back the template
        # unchanged.
        return FlexiChains.FlexiChain{Symbol}(
            FlexiChains.niters(chain), FlexiChains.nchains(chain),
            Dict{FlexiChains.ParameterOrExtra{<:Symbol}, Matrix}())
    end
    return FlexiChains.map_parameters(chain) do vn
        haskey(lookup, vn) || throw(ArgumentError(
            "chain parameter $vn is not one of $(typeof(obj))'s estimated " *
            "rows at prefix $(repr(prefix))"))
        lookup[vn]
    end
end

# `distribution_params`/`readback`/`readback_draws` for a `VarName`-keyed
# chain (e.g. sampled from `as_turing` with `chain_type =
# FlexiChains.VNChain`): convert onto the dotted `Symbol` naming via
# `_to_symbol_chain`, then delegate to the `Symbol`-keyed method. `prefix`
# matches the `prefix` `as_turing` was called with (default `:d`); every other
# keyword forwards.
function distribution_params(obj, chain::FlexiChains.FlexiChain{<:VarName};
        prefix::Symbol = :d, kwargs...)
    return distribution_params(
        obj, _to_symbol_chain(obj, chain, prefix); kwargs...)
end

function readback(obj, chain::FlexiChains.FlexiChain{<:VarName};
        prefix::Symbol = :d, kwargs...)
    return readback(obj, _to_symbol_chain(obj, chain, prefix); kwargs...)
end

function readback_draws(obj, chain::FlexiChains.FlexiChain{<:VarName};
        prefix::Symbol = :d, kwargs...)
    return readback_draws(obj, _to_symbol_chain(obj, chain, prefix); kwargs...)
end

end # module DistributionsInferenceDynamicPPLFlexiChainsExt
