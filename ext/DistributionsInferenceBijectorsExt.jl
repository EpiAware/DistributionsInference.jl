# DistributionsInference x Bijectors: the unconstrained <-> constrained
# transform. Each estimated row carries its constraint in its own prior, so
# `bijector(prior)` per row gives the flat transform with no domain table.
module DistributionsInferenceBijectorsExt

using DistributionsInference: DistributionsInference, FitLogDensity,
                              logdensity
using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian

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
    priors = prob.flat_priors
    length(z) == length(priors) || throw(DimensionMismatch(
        "unconstrained vector has length $(length(z)) but $(prob.obj) has " *
        "$(length(priors)) estimated parameter(s)"))
    xs_and_logj = map(i -> _row_transform(priors[i], i, z[i]), eachindex(z))
    x = [xi for (xi, _) in xs_and_logj]
    logjac = isempty(xs_and_logj) ? zero(eltype(z)) : sum(last, xs_and_logj)
    return x, logjac
end

# The forward direction: no log-Jacobian, since a caller mapping known
# parameter values onto the sampling scale wants the point, not its density.
function DistributionsInference.to_unconstrained(
        prob::FitLogDensity, x::AbstractVector)
    priors = prob.flat_priors
    length(x) == length(priors) || throw(DimensionMismatch(
        "constrained vector has length $(length(x)) but $(prob.obj) has " *
        "$(length(priors)) estimated parameter(s)"))
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
    return DistributionsInference.reconstruct(prob.obj, x)
end

end # module
