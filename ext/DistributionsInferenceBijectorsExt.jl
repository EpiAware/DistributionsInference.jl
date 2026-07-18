# DistributionsInference x Bijectors: the prior-driven unconstrained <->
# constrained transform for the PPL-neutral engine. Each ESTIMATED flat row's
# constraint is carried by its prior itself, so `bijector(prior)` per row
# gives the flat transform with no bespoke domain table. Loaded only when
# Bijectors is available. Ported from ComposedDistributions'
# `ComposedDistributionsBijectorsExt` (ComposedDistributions#185), generalised
# from a composed tree's `flat_priors` (which may carry a `CentredPoolPrior`
# marker row) to the row-based fit protocol's `FitLogDensity` — a row with no
# per-row prior (`prior === nothing`, scored instead through `extra_logprior`)
# has no family this generic extension can read a bijector off, so it is
# rejected rather than special-cased, mirroring the `DynamicPPL` extension's
# `_validate_turing_rows`.
module DistributionsInferenceBijectorsExt

using DistributionsInference: DistributionsInference, FitLogDensity
using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian

# The bijector for one ESTIMATED row's prior. A row with no per-row prior
# (an object-dependent prior, scored instead through `extra_logprior`) has no
# family to read a bijector off, so it is rejected with a clear pointer at the
# row's position rather than a bare `MethodError` from `bijector(nothing)`.
function _row_bijector(prior, i)
    prior === nothing && throw(ArgumentError(
        "to_constrained has no bijector for estimated row $i: its prior is " *
        "`nothing` (an object-dependent prior scored through " *
        "`extra_logprior`, e.g. a hierarchical population term); a type " *
        "with such a row needs its own `to_constrained` method"))
    return bijector(prior)
end

# The per-row inverse bijectors (unconstrained -> constrained), one per
# ESTIMATED flat parameter, in table-row order. `FitLogDensity` already
# carries `flat_priors` (flattened once at construction), so no table walk is
# needed here.
function _inverse_bijectors(prob::FitLogDensity)
    return [inverse(_row_bijector(prior, i))
            for (i, prior) in enumerate(prob.flat_priors)]
end

# `to_constrained(prob, z)`: push each unconstrained coordinate through its
# row's inverse bijector, accumulating the log-Jacobian. Every transform here
# is univariate (one scalar prior per row), so the estimated dimension is
# unchanged and the map is element-wise; the total log-Jacobian is the sum of
# the per-row terms.
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

end # module
