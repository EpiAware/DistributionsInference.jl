module DistributionsInferenceDynamicPPLExt

# DistributionsInference x DynamicPPL: `as_turing(obj, data)` builds a
# DynamicPPL model over a fittable object's estimated rows, wrapping the
# `as_logdensity` codec (docstring in `src/turing.jl`). Each row's dotted
# `name` becomes the `VarName` `<prefix>.<name>`, sampled from its own prior.
#
# `_row_varname` is the naming contract shared with the `VarName`-keyed
# readback in `DistributionsInferenceDynamicPPLFlexiChainsExt` (DI#4, the
# two-extension split); it is declared as a stub in `src/turing.jl` so that
# extension reaches it by dispatch rather than through a sibling extension's
# module.

using DistributionsInference: DistributionsInference, FitLogDensity,
                              as_logdensity, estimated_rows, reconstruct,
                              extra_logprior, _check_generic_fields
import DistributionsInference: as_turing, _row_varname
using DynamicPPL: DynamicPPL, @model, NamedDist, VarName

# There is no public constructor for a runtime dotted optic, so the two
# primitives `@varname` lowers to are reached through `AbstractPPL` itself:
# `Property{sym}(child)` for a `.sym` access and `Iden()` for the leaf.
const _AbstractPPL = parentmodule(VarName)
const _Property = _AbstractPPL.Property
const _Iden = _AbstractPPL.Iden

# Build a row's `VarName`, e.g. prefix `:d` and `Symbol("onset.shape")` give
# `"d.onset.shape"`. The optic is folded outermost-property-first (`reverse`)
# so the earliest segment renders nearest the prefix.
function _row_varname(prefix::Symbol, name::Symbol)
    segs = Tuple(Symbol(s) for s in split(string(name), "."))
    optic = foldl((acc, s) -> _Property{s}(acc), reverse(segs); init = _Iden())
    return VarName{prefix}(optic)
end

# A row with `prior === nothing` has no distribution to sample it from.
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

# `NamedDist` sets the site name to the readback name regardless of the LHS.
# Priors go through `~` and everything else through `@addlogprob!`, so there
# is no double counting and the total equals `logdensity(prob, θ)`. `θ` has an
# abstract element type so AD values flow through `reconstruct` unchanged.
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
