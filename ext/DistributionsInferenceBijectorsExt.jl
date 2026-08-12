# DistributionsInference x Bijectors: the unconstrained <-> constrained
# transform. Each estimated row carries its constraint in its own prior, so
# `bijector(prior)` per row gives the flat transform with no domain table.
module DistributionsInferenceBijectorsExt

using DistributionsInference: DistributionsInference, FitLogDensity,
                              logdensity, template, flat_priors
using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian
using Distributions: MvNormal
using FlexiChains: FlexiChains
using LinearAlgebra: isposdef
using Random: Random
using Statistics: mean, cov

function _row_bijector(prior, i)
    prior === nothing && throw(ArgumentError(
        "no bijector for estimated row $i: its prior is " *
        "`nothing` (an object-dependent prior scored through " *
        "`extra_logprior`, e.g. a hierarchical population term); a type " *
        "with such a row needs its own `to_constrained`/`to_unconstrained` " *
        "methods"))
    return bijector(prior)
end

# One row's unconstrained -> constrained step, returning `(x_i, logjac_i)`.
# The bijector is built and consumed here rather than materialised into a
# per-row array: `flat_priors` is abstractly typed whenever a tree mixes
# prior families, and Enzyme's reverse mode cannot build a shadow for such an
# array (DI#33).
function _row_transform(prior, i, zi)
    return with_logabsdet_jacobian(inverse(_row_bijector(prior, i)), zi)
end

# Every transform is univariate (one scalar prior per row), so the map is
# element-wise and the log-Jacobian is the sum of the per-row terms.
function DistributionsInference.to_constrained(
        prob::FitLogDensity, z::AbstractVector)
    priors = flat_priors(prob)
    length(z) == length(priors) || throw(DimensionMismatch(
        "unconstrained vector has length $(length(z)) but $(template(prob)) " *
        "has $(length(priors)) estimated parameter(s)"))
    xs_and_logj = map(i -> _row_transform(priors[i], i, z[i]), eachindex(z))
    x = [xi for (xi, _) in xs_and_logj]
    logjac = isempty(xs_and_logj) ? zero(eltype(z)) : sum(last, xs_and_logj)
    return x, logjac
end

# The forward direction: no log-Jacobian, since a caller mapping known
# parameter values onto the sampling scale wants the point, not its density.
function DistributionsInference.to_unconstrained(
        prob::FitLogDensity, x::AbstractVector)
    priors = flat_priors(prob)
    length(x) == length(priors) || throw(DimensionMismatch(
        "constrained vector has length $(length(x)) but $(template(prob)) " *
        "has $(length(priors)) estimated parameter(s)"))
    return [_row_bijector(priors[i], i)(x[i]) for i in eachindex(x)]
end

# Composes `to_constrained` with the core `logdensity` (DI#46).
function DistributionsInference.logdensity_to_objective(prob::FitLogDensity)
    return function (z::AbstractVector)
        x, logjac = DistributionsInference.to_constrained(prob, z)
        return -(logdensity(prob, x) + logjac)
    end
end

# The last step of a point fit: an optimiser's minimiser back to an object.
function DistributionsInference.objective_to_distribution(
        prob::FitLogDensity, z::AbstractVector)
    x, _ = DistributionsInference.to_constrained(prob, z)
    return DistributionsInference.reconstruct(template(prob), x)
end

# `inference_to_parameter_distribution`'s joint Gaussian: reuses
# `inference_to_parameters` for the exact per-row draws — its `NamedTuple`
# preserves `estimated_rows(obj)` row order, the same order `flat_priors`/
# `to_unconstrained` build their per-row transform from, so no re-ordering is
# needed between the two.
#
# `data` is `nothing`: `to_unconstrained` only reads `prob`'s `flat_priors`
# (built from `obj`'s own priors) and never `prob.data`/`prob.scored_data`,
# so no real data is needed to build the `FitLogDensity` this borrows purely
# for its transform. This is LOAD-BEARING: `FitLogDensity` construction
# (`engine.jl`) runs `_prepare_scored_data(obj, data, loglik)`, which for the
# default `loglik` dispatches to `_prepare_default_loglik_data(obj, data)`
# (#115) — checked against `nothing` here because only
# `ComposedDistributions` overrides it, on `data::AbstractVector{<:NamedTuple}`,
# which `nothing` never matches, so every path falls through to the
# `(obj, data) = data` no-op and `nothing` passes through unchanged. Anyone
# adding a `_prepare_scored_data`/`_prepare_default_loglik_data` override
# that reads `data` unconditionally (rather than dispatching away from a
# non-matching shape) breaks this call.
function DistributionsInference.inference_to_parameter_distribution(
        obj, chain::FlexiChains.FlexiChain;
        draws = nothing, rng = Random.default_rng())
    nt = DistributionsInference.inference_to_parameters(
        obj, chain; draws = draws, rng = rng)
    isempty(nt) && throw(ArgumentError(
        "`inference_to_parameter_distribution` needs at least one " *
        "estimated parameter; `obj` estimates nothing"))
    cols = values(nt)
    n = length(first(cols))
    n == 0 && throw(ArgumentError(
        "`inference_to_parameter_distribution` needs at least one " *
        "selected draw; the resolved `draws` selection is empty"))
    dim = length(cols)
    # A Gaussian fitted to `n <= dim` draws gives a singular (at best
    # rank-deficient) sample covariance, which `MvNormal` would otherwise
    # reject with a bare `PosDefException` from inside PDMats, naming
    # neither this function nor the cause.
    n > dim || throw(ArgumentError(
        "`inference_to_parameter_distribution` needs more selected draws " *
        "than estimated parameters to fit a non-singular covariance: got " *
        "$n draw(s) for $dim parameter(s)"))
    T = promote_type(map(eltype, cols)...)

    prob = DistributionsInference.distribution_to_logdensity(obj, nothing)
    x = Vector{T}(undef, dim)
    z = Matrix{T}(undef, dim, n)
    for j in 1:n
        for (i, col) in enumerate(cols)
            x[i] = col[j]
        end
        z[:, j] = DistributionsInference.to_unconstrained(prob, x)
    end
    mu = vec(mean(z; dims = 2))
    Sigma = cov(z; dims = 2)
    # Enough draws is necessary but not sufficient: exactly collinear rows
    # (one parameter a deterministic function of another on the
    # unconstrained scale) give a rank-deficient covariance at any draw
    # count, and `MvNormal` then fails inside PDMats' Cholesky with a
    # `PosDefException` naming neither this function nor the cause.
    isposdef(Sigma) || throw(ArgumentError(
        "`inference_to_parameter_distribution` could not fit a Gaussian: " *
        "the covariance of the unconstrained draws is not positive " *
        "definite. That happens when two estimated parameters are " *
        "collinear on the unconstrained scale (one a deterministic " *
        "function of another), or when a parameter does not vary across " *
        "the selected draws. Estimated parameters: " *
        "$(collect(keys(nt))); selected draws: $n."))
    return MvNormal(mu, Sigma)
end

function DistributionsInference.inference_to_parameter_distribution(
        obj, raw::Union{AbstractMatrix, AbstractVector{<:AbstractVector}};
        draws = nothing, nchains::Int = 1, rng = Random.default_rng())
    return DistributionsInference.inference_to_parameter_distribution(
        obj,
        DistributionsInference.draws_to_chain(obj, raw; nchains = nchains);
        draws = draws, rng = rng)
end

end # module
