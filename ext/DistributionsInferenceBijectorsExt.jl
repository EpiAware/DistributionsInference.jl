# DistributionsInference x Bijectors: the unconstrained <-> constrained
# transform. Each estimated row carries its constraint in its own prior, so
# `bijector(prior)` per row gives the flat transform with no domain table.
module DistributionsInferenceBijectorsExt

using DistributionsInference: DistributionsInference, FitLogDensity,
                              logdensity
using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian

function _row_bijector(prior, i)
    prior === nothing && throw(ArgumentError(
        "to_constrained has no bijector for estimated row $i: its prior is " *
        "`nothing` (an object-dependent prior scored through " *
        "`extra_logprior`, e.g. a hierarchical population term); a type " *
        "with such a row needs its own `to_constrained` method"))
    return bijector(prior)
end

# The per-row inverse bijectors (unconstrained -> constrained), in row order.
function _inverse_bijectors(prob::FitLogDensity)
    return [inverse(_row_bijector(prior, i))
            for (i, prior) in enumerate(prob.flat_priors)]
end

# Every transform is univariate (one scalar prior per row), so the map is
# element-wise and the log-Jacobian is the sum of the per-row terms.
function DistributionsInference.to_constrained(
        prob::FitLogDensity, z::AbstractVector)
    binvs = _inverse_bijectors(prob)
    length(z) == length(binvs) || throw(DimensionMismatch(
        "unconstrained vector has length $(length(z)) but $(prob.obj) has " *
        "$(length(binvs)) estimated parameter(s)"))
    xs_and_logj = map((b, zi) -> with_logabsdet_jacobian(b, zi), binvs, z)
    x = [xi for (xi, _) in xs_and_logj]
    logjac = isempty(xs_and_logj) ? zero(eltype(z)) : sum(last, xs_and_logj)
    return x, logjac
end

# Composes `to_constrained` with the core `logdensity` (DI#46).
function DistributionsInference.as_optimisation_objective(prob::FitLogDensity)
    return function (z::AbstractVector)
        x, logjac = DistributionsInference.to_constrained(prob, z)
        return -(logdensity(prob, x) + logjac)
    end
end

end # module
