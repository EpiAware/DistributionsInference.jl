# # [Fitting a custom distribution](@id custom-distribution)
#
# A distribution written by hand becomes fittable by answering two questions:
# what are its scalar parameters, and how does it rebuild itself from a flat
# vector.
# Nothing else changes, and no probabilistic programming language is involved.
#
# This tutorial writes those two methods for a gamma-backed delay, samples the
# posterior with a plain Metropolis sampler, reads the draws back onto the
# distribution, swaps the likelihood for a survival one, and finds a
# maximum-a-posteriori point with an external optimiser.

using DistributionsInference, Distributions, Random
using FlexiChains: FlexiChains

# ## Declaring the parameters
#
# `parameter_rows` returns one row per scalar parameter, carrying its name,
# current value, prior and support.
# A row with a prior is estimated; a row with `prior = nothing` is held fixed.
# `reconstruct` is the one place the type says how to rebuild itself from the
# flat vector the engine works over.

struct ToyDelay{T <: Real}
    shape::T
    scale::T
end

Distributions.logpdf(d::ToyDelay, y::Real) = logpdf(Gamma(d.shape, d.scale), y)

function DistributionsInference.parameter_rows(d::ToyDelay)
    return [
        (name = :shape, value = d.shape,
            prior = LogNormal(log(2.0), 0.2), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing,
            support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ToyDelay, x::AbstractVector)
    return ToyDelay(x[1], oftype(x[1], d.scale))
end

delay = ToyDelay(2.0, 1.0)
data = [1.5, 2.0, 3.2, 1.8, 2.6]

parameter_rows(delay)

# The fields are typed `T <: Real` and the fixed `scale` is rebuilt through
# `oftype`, so `reconstruct` returns whatever number type it is handed.
# A gradient-based sampler differentiates through this call and passes a dual
# number rather than a `Float64`; a concretely typed field would reject it.

DistributionsInference.reconstruct(delay, [1 // 2])

# ## Scoring a parameter vector
#
# [`distribution_to_logdensity`](@ref) packages the distribution and the data
# into a log-density over the estimated rows alone, and
# [`flat_dimension`](@ref DistributionsInference.flat_dimension) counts them.

prob = distribution_to_logdensity(delay, data)
DistributionsInference.flat_dimension(delay)

# [`logdensity`](@ref DistributionsInference.logdensity) adds the prior's
# log-density at that value to the data likelihood of the distribution rebuilt
# there.

DistributionsInference.logdensity(prob, [2.0])

# ## Sampling with no probabilistic programming language
#
# `prob` implements the `LogDensityProblems` interface, so any consumer of that
# interface can drive it.
# `AdvancedMH`'s random-walk Metropolis is the smallest one to reach for; a
# gradient backend added through `LogDensityProblemsAD` opens up AdvancedHMC
# the same way.
# The wrapper below returns `-Inf` off `shape`'s positive support, which a
# random-walk proposal does not respect on its own.

using AdvancedMH
using LinearAlgebra: I

model = AdvancedMH.DensityModel() do x
    any(<=(0), x) ? -Inf : DistributionsInference.logdensity(prob, x)
end
sampler = RWMH(MvNormal(zeros(1), 0.05^2 * I))
transitions = sample(Xoshiro(1), model, sampler, 2000;
    param_names = ["shape"], progress = false)
draws = [t.params for t in transitions][1001:end]
length(draws)

# ## Reading the fit back onto the distribution
#
# [`to_flexichain`](@ref) keys the raw draws by the estimated rows' dotted
# names, the naming contract every readback verb here uses.

chain = to_flexichain(delay, draws)

# [`point_estimate`](@ref) reduces the chain straight back to a `ToyDelay`,
# posterior mean by default.

fitted = point_estimate(delay, chain)
fitted.shape

# The fixed row came back untouched, at its template value.

fitted.scale

# [`distribution_params`](@ref) is the params-first primitive underneath: the
# same reduction keyed by dotted name, before the object is rebuilt.

distribution_params(delay, chain)

# [`readback_draws`](@ref) keeps every draw instead of reducing them, which is
# what a per-draw posterior-predictive summary needs.

all_fitted = readback_draws(delay, chain)
quantile([mean(Gamma(d.shape, d.scale)) for d in all_fitted], [0.025, 0.975])

# ## Swapping the likelihood
#
# `loglik` is a reducer `(obj, data) -> Real`, not a fixed sum of `logpdf`.
# Right-censored records, known only to exceed a bound, score through `logccdf`
# instead.

function survival_loglik(obj, records)
    return sum(y -> logccdf(Gamma(obj.shape, obj.scale), y), records)
end
bounds = [1.0, 1.5, 2.0]
survival_prob = distribution_to_logdensity(delay, bounds;
    loglik = survival_loglik)
DistributionsInference.logdensity(survival_prob, [2.0])

# Everything above works unchanged on `survival_prob`: the parameter
# inventory, the priors and the readback do not know which reducer scored the
# data.

survival_model = AdvancedMH.DensityModel() do x
    any(<=(0), x) ? -Inf : DistributionsInference.logdensity(survival_prob, x)
end
survival_transitions = sample(Xoshiro(1), survival_model, sampler, 2000;
    param_names = ["shape"], progress = false)
survival_draws = [t.params for t in survival_transitions][1001:end]
point_estimate(delay, to_flexichain(delay, survival_draws)).shape

# A gradient-free sampler is the right choice here: `logccdf` for a `Gamma` has
# no `ForwardDiff` rule, so this reducer cannot be handed to `NUTS`.
# The same applies to a reducer that draws random replicates internally to
# marginalise a latent variable, which is noisier as well as usually
# non-differentiable; raise its replicate count until the posterior summary
# stops moving.

# ## A point estimate instead of a posterior
#
# The objective is an unnormalised log-density, so an external optimiser can
# find its mode directly.
# [`logdensity_to_objective`](@ref) is the wiring only: the unconstrained
# transform, the negated objective and
# [`reconstruct`](@ref DistributionsInference.reconstruct), once `Bijectors` is
# loaded.
# The optimiser stays external.

using Bijectors, Optim

f = logdensity_to_objective(prob)
res = optimize(f, zeros(DistributionsInference.flat_dimension(delay)), LBFGS())
z_hat = Optim.minimizer(res)

# `z_hat` is on the unconstrained scale.
# [`to_constrained`](@ref DistributionsInference.to_constrained) and
# [`reconstruct`](@ref DistributionsInference.reconstruct) push it back, the
# same readback path every sampler above reconstructs through.

x_hat, _ = DistributionsInference.to_constrained(prob, z_hat)
map_fit = DistributionsInference.reconstruct(prob.obj, x_hat)
map_fit.shape

# That is the maximum-a-posteriori point, because `logdensity` always adds an
# estimated row's own prior.
# Maximum likelihood needs a prior flat enough to leave the likelihood alone,
# rather than a different reducer.

struct ToyDelayML{T <: Real}
    shape::T
    scale::T
end

function Distributions.logpdf(d::ToyDelayML, y::Real)
    return logpdf(Gamma(d.shape, d.scale), y)
end

function DistributionsInference.parameter_rows(d::ToyDelayML)
    return [
        (name = :shape, value = d.shape,
            prior = LogNormal(0.0, 100.0), support = (0.0, Inf)),
        (name = :scale, value = d.scale, prior = nothing, support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(d::ToyDelayML, x::AbstractVector)
    return ToyDelayML(x[1], oftype(x[1], d.scale))
end

ml_delay = ToyDelayML(2.0, 1.0)
ml_prob = distribution_to_logdensity(ml_delay, data)
ml_res = optimize(logdensity_to_objective(ml_prob),
    zeros(DistributionsInference.flat_dimension(ml_delay)), LBFGS())
ml_x, _ = DistributionsInference.to_constrained(
    ml_prob, Optim.minimizer(ml_res))
DistributionsInference.reconstruct(ml_delay, ml_x).shape

# The data pull the shape above the prior mean of 2 either way, and the diffuse
# prior pulls it back less.

# ## Next
#
# - [Sampling with Turing](@ref turing-sampling) fits this same distribution
#   through `DynamicPPL` and reads the chain back with the same calls.
# - [Fitting a composed distribution](@ref composed-distributions) runs both
#   routes against a tree built by `ComposedDistributions`, with no protocol
#   methods written at all.
# - [Public API](@ref public-api) lists the rest of the protocol
#   (`with_priors`, `estimated_rows`, `extra_logprior`).
