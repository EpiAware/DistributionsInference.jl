module DistributionsInferenceDynamicPPLFlexiChainsExt

# DistributionsInference x DynamicPPL x FlexiChains: the `VarName`-keyed chain
# readback (DI#4). A chain sampled from `as_turing` is keyed by `VarName`, so
# rename it onto the estimated rows' dotted `Symbol` names and hand it to the
# `Symbol`-keyed readback in `DistributionsInferenceFlexiChainsExt`.
# Docstrings live on the stubs in `src/readback.jl`.

using DistributionsInference: DistributionsInference, estimated_rows,
                              _row_varname
import DistributionsInference: distribution_params, readback, readback_draws
using DynamicPPL: VarName
using FlexiChains: FlexiChains

# Rename a `VarName`-keyed chain's parameters onto the estimated rows' dotted
# `Symbol` names, so it matches what `to_flexichain` would have built.
# `_row_varname` is the DynamicPPL extension's method, reached by dispatch on
# the core stub, so the two extensions cannot drift apart on the naming.
function _to_symbol_chain(obj, chain::FlexiChains.FlexiChain{<:VarName},
        prefix::Symbol)
    lookup = Dict(_row_varname(prefix, row.name) => row.name
    for row in estimated_rows(obj))
    if isempty(FlexiChains.parameters(chain))
        # `FlexiChains.map_parameters` infers `Union{}` as the key type on a
        # chain with zero parameters, and a `FlexiChain{Union{}}` stack
        # overflows on its own reconstruction, so build the empty chain here.
        # The guard is on the chain, not on `obj`'s rows: an `obj` with
        # estimated rows against an empty chain is a mismatch and must still
        # reach the readback's `has_parameter` check.
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

# `prefix` must match the one `as_turing` was called with; every other keyword
# forwards to the `Symbol`-keyed method.
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
