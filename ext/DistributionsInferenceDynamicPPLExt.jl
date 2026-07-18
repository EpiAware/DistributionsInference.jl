module DistributionsInferenceDynamicPPLExt

# DistributionsInference x DynamicPPL: `as_turing(obj, data)` builds a
# DynamicPPL model over a fittable object's ESTIMATED parameters (the fit
# protocol's `estimated_rows`), a light wrapper on the `as_logdensity` codec
# (declared with its docstring in `src/turing.jl`). Loaded when DynamicPPL
# alone is available, so the core package stays Turing-free. Ported from
# ComposedDistributions' `ComposedDistributionsDynamicPPLExt`
# (ComposedDistributions#185), generalised from a composed tree's nested edge
# path to the row-based fit protocol: each estimated row's dotted `name`
# (e.g. `Symbol("onset.shape")`) becomes the DynamicPPL `VarName`
# `<prefix>.onset.shape`, sampled from its own `prior` at a named `~` site,
# with the data likelihood plus `extra_logprior` added via `@addlogprob!`
# from the codec's `reconstruct`.
#
# This same file hosts the `VarName`-keyed FlexiChains readback (DI#4): a
# chain sampled from `as_turing` (keyed by `VarName`, e.g. via `chain_type =
# FlexiChains.VNChain`) is renamed onto the estimated rows' dotted `Symbol`
# names and handed to the core `readback`/`readback_draws`
# (`src/readback.jl`), so the DynamicPPL naming contract needs no separate
# readback surface — the docstrings on those two functions stay the single
# source of truth (no split-across-modules duplicate, mirroring
# ComposedDistributionsFlexiChainsExt's `update` convention).

using DistributionsInference: DistributionsInference, FitLogDensity,
                              as_logdensity, estimated_rows, reconstruct,
                              extra_logprior
import DistributionsInference: as_turing, readback, readback_draws
using DynamicPPL: DynamicPPL, @model, NamedDist, VarName
using FlexiChains: FlexiChains

# `AbstractPPL` (re-exported through DynamicPPL/FlexiChains) owns the
# `VarName` optic types. There is no public constructor for a runtime dotted
# optic, so the two optic primitives are reached through the parent module:
# `Property{sym}(child)` for a `.sym` access and `Iden()` for the leaf. These
# are the same primitives DynamicPPL's own `@varname` lowers to (mirrors
# ComposedDistributionsDynamicPPLExt's `_dotted_varname`).
const _AbstractPPL = parentmodule(VarName)
const _Property = _AbstractPPL.Property
const _Iden = _AbstractPPL.Iden

# Build the `VarName` an estimated row's `~` site carries: the `prefix`
# symbol then the row's dotted `name` split on `.` into its segments, so
# `string(vn)` is `"<prefix>.<segs...>"` (e.g. prefix `:d`, name
# `Symbol("onset.shape")` -> `"d.onset.shape"`). The optic is built
# outermost-property-first (`reverse`) so the earliest segment renders
# nearest the prefix. The same construction, run in reverse (row name ->
# VarName), is what `_to_symbol_chain` below matches a readback chain's keys
# against.
function _row_varname(prefix::Symbol, name::Symbol)
    segs = Tuple(Symbol(s) for s in split(string(name), "."))
    optic = foldl((acc, s) -> _Property{s}(acc), reverse(segs); init = _Iden())
    return VarName{prefix}(optic)
end

# Reject an estimated row with no fixed `~` prior (`prior === nothing`,
# scored instead through `extra_logprior` — an object-dependent prior term;
# see `parameter_rows`): it has no distribution to sample it from under
# DynamicPPL. Generalises ComposedDistributionsDynamicPPLExt's centred-pool
# rejection (`_reject_pools`) to the protocol's own row schema, so this
# extension needs no ComposedDistributions-specific `Pool` knowledge.
function _validate_turing_rows(obj)
    rows = estimated_rows(obj)
    missing_prior = [row.name for row in rows if row.prior === nothing]
    isempty(missing_prior) || throw(ArgumentError(
        "as_turing does not support estimated parameter(s) $missing_prior " *
        "with no fixed `~` prior (scored instead through `extra_logprior`, " *
        "an object-dependent prior term whose sampling path does not exist " *
        "yet in DynamicPPL). Sample with `as_logdensity(obj, data)` + " *
        "LogDensityProblemsAD (the LogDensityProblems extension) instead."))
    return rows
end

# The model: sample each estimated row from its prior at its dotted VarName
# (a `NamedDist`, so the site name is the readback name regardless of the
# LHS), then add the data likelihood and `extra_logprior` with
# `@addlogprob!` from the codec's reconstruction. Priors via `~`, everything
# else via `@addlogprob!`, so no double counting; the total equals
# `logdensity(prob, θ)`. `θ` has an abstract element type so a sampled/AD
# value (a `Dual`/tracked number) flows through `reconstruct` unchanged.
@model function _fit_turing_model(prob::FitLogDensity, vns)
    fp = prob.flat_priors
    n = length(fp)
    θ = Vector{Real}(undef, n)
    for i in 1:n
        param ~ NamedDist(fp[i], vns[i])
        θ[i] = param
    end
    obj = reconstruct(prob.obj, θ)
    DynamicPPL.@addlogprob! extra_logprior(prob.obj, obj, θ)
    DynamicPPL.@addlogprob! prob.loglik(obj, prob.data)
    return obj
end

function as_turing(obj, data;
        prefix::Symbol = :d, loglik = DistributionsInference._default_loglik)
    rows = _validate_turing_rows(obj)
    prob = as_logdensity(obj, data; loglik = loglik)
    vns = [_row_varname(prefix, row.name) for row in rows]
    return _fit_turing_model(prob, vns)
end

# Rename a `VarName`-keyed chain's parameters onto the estimated rows' dotted
# `Symbol` names, so it matches what the core `to_flexichain` would have
# built, and the existing `Symbol`-keyed readback machinery
# (`_flat_from_chain`/`_chain_column` in `src/readback.jl`) reads it
# unchanged. A chain parameter that is not one of `obj`'s estimated rows'
# `VarName`s signals a chain that was not sampled from `as_turing(obj, ...)`
# at this `prefix` (wrong prefix, or a mismatched template), so it errors
# rather than silently dropping the column.
function _to_symbol_chain(obj, chain::FlexiChains.FlexiChain{<:VarName},
        prefix::Symbol)
    lookup = Dict(_row_varname(prefix, row.name) => row.name
    for row in estimated_rows(obj))
    if isempty(lookup)
        # `FlexiChains.map_parameters` infers the wrong key type on a chain
        # with zero parameters (an empty `Set` there resolves to `Union{}`,
        # not `Symbol`, and building a `FlexiChain{Union{}}` stack-overflows
        # on its own `NamedTuple` reconstruction): build a fresh empty
        # `Symbol`-keyed chain directly instead, mirroring `to_flexichain`'s
        # own 0-estimated construction in `src/readback.jl`.
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

# `readback`/`readback_draws` for a `VarName`-keyed chain (e.g. sampled from
# `as_turing` with `chain_type = FlexiChains.VNChain`): convert onto the
# dotted `Symbol` naming via `_to_symbol_chain`, then delegate to the core
# `Symbol`-keyed method. `prefix` matches the `prefix` `as_turing` was called
# with (default `:d`); every other keyword forwards to the core method.
function readback(obj, chain::FlexiChains.FlexiChain{<:VarName};
        prefix::Symbol = :d, kwargs...)
    return readback(obj, _to_symbol_chain(obj, chain, prefix); kwargs...)
end

function readback_draws(obj, chain::FlexiChains.FlexiChain{<:VarName};
        prefix::Symbol = :d, kwargs...)
    return readback_draws(obj, _to_symbol_chain(obj, chain, prefix); kwargs...)
end

end # module DistributionsInferenceDynamicPPLExt
