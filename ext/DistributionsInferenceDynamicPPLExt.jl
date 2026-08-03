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
# The `VarName`-keyed chain readback that pairs with this model (DI#4) needs
# `FlexiChains` as well as `DynamicPPL`, so it lives in its own extension,
# `DistributionsInferenceDynamicPPLFlexiChainsExt`. Keeping the split means
# `as_turing` stays available to a project that loads `DynamicPPL` alone: the
# model itself never touches a chain type. The `VarName` naming contract the
# two extensions share is `_row_varname` below, whose method this extension
# owns (it is declared as a stub in `src/turing.jl` so the readback extension
# reaches it by ordinary dispatch rather than through a sibling extension's
# module).

using DistributionsInference: DistributionsInference, FitLogDensity,
                              as_logdensity, estimated_rows, reconstruct,
                              extra_logprior, _check_generic_fields
import DistributionsInference: as_turing, _row_varname
using DynamicPPL: DynamicPPL, @model, NamedDist, VarName

# `AbstractPPL` (re-exported through DynamicPPL) owns the `VarName` optic
# types. There is no public constructor for a runtime dotted optic, so the
# two optic primitives are reached through the parent module:
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
# VarName), is what the readback extension's `_to_symbol_chain` matches a
# chain's keys against — hence the core stub this method fills in.
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
    _check_generic_fields(typeof(prob.obj), prob.concrete_fields, θ)
    obj = reconstruct(prob.obj, θ)
    DynamicPPL.@addlogprob! extra_logprior(prob.obj, obj, θ, prob.extra_state)
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

end # module DistributionsInferenceDynamicPPLExt
