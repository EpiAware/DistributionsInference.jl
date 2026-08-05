# # [Fitting a custom distribution](@id custom-distribution)
#
# A distribution written by hand becomes fittable by answering two questions:
# what are its scalar parameters, and how does it rebuild itself from a flat
# vector.
# Nothing else changes.
#
# This tutorial writes those two methods for a Weibull-backed delay, samples the
# posterior with a plain Metropolis sampler and reads the draws back onto the
# distribution.
# It then swaps the likelihood for a survival one and finds a
# maximum-a-posteriori point with an external optimiser.

using DistributionsInference, Distributions, Random
using FlexiChains: FlexiChain, Parameter

# ## Declaring the parameters
#
# [`parameter_rows`](@ref) returns one row per scalar parameter, carrying its
# name, current value, prior and support.
# A row with a prior is estimated; a row with `prior = nothing` is held fixed.
# [`reconstruct`](@ref DistributionsInference.reconstruct) is the one place the
# type says how to rebuild itself from the flat vector the engine works over.

struct ToyDelay{T <: Real}
    shape::T
    scale::T
end

function Distributions.logpdf(d::ToyDelay, y::Real)
    return logpdf(Weibull(d.shape, d.scale), y)
end

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

delay = ToyDelay(2.0, 2.5)
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

# ## Sampling the log-density directly
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
# The readback reads a `FlexiChain` keyed by the estimated rows' dotted names,
# the naming contract every readback verb here uses.
# Building that chain out of a sampler's raw draws is FlexiChains' own
# constructor rather than a step this package owns.

niter = length(draws)
chain = FlexiChain{Symbol}(niter, 1,
    Dict(Parameter(:shape) => reshape(first.(draws), niter, 1)))

# [`point_estimate`](@ref) reduces the chain straight back to a `ToyDelay`,
# posterior mean by default.

fitted = point_estimate(delay, chain)
fitted.shape

# The fixed row came back at its template value.

fitted.scale

# [`distribution_params`](@ref) is the params-first primitive underneath, the
# same reduction keyed by dotted name before the object is rebuilt.

distribution_params(delay, chain)

# [`distribution_draws`](@ref) keeps every draw instead of reducing them, for a
# per-draw posterior-predictive summary.

all_fitted = distribution_draws(delay, chain)
quantile([mean(Weibull(d.shape, d.scale)) for d in all_fitted], [0.025, 0.975])

# ## Swapping the likelihood
#
# `loglik` is any reducer `(obj, data) -> Real`.
# Right-censored records, known only to exceed a bound, score through `logccdf`
# rather than `logpdf`.

function survival_loglik(obj, records)
    return sum(y -> logccdf(Weibull(obj.shape, obj.scale), y), records)
end
bounds = [1.0, 1.5, 2.0]
survival_prob = distribution_to_logdensity(delay, bounds;
    loglik = survival_loglik)
DistributionsInference.logdensity(survival_prob, [2.0])

# The parameter inventory, the priors and the readback never see the reducer,
# so everything above works unchanged on `survival_prob`.

survival_model = AdvancedMH.DensityModel() do x
    any(<=(0), x) ? -Inf : DistributionsInference.logdensity(survival_prob, x)
end
survival_transitions = sample(Xoshiro(1), survival_model, sampler, 2000;
    param_names = ["shape"], progress = false)
survival_draws = [t.params for t in survival_transitions][1001:end]
survival_chain = FlexiChain{Symbol}(length(survival_draws), 1,
    Dict(Parameter(:shape) => reshape(
        first.(survival_draws), length(survival_draws), 1)))
point_estimate(delay, survival_chain).shape

# `logccdf` for a `Weibull` differentiates, so
# [Sampling with Turing](@ref turing-sampling) hands this same reducer to
# `NUTS`.
# A reducer that draws random replicates internally to marginalise a latent
# variable is noisy as well as usually non-differentiable, so keep it on a
# gradient-free sampler and raise its replicate count until the posterior
# summary stops moving.

# ## A point estimate instead of a posterior
#
# The objective is an unnormalised log-density, so an optimiser can find its
# mode directly.
# [`optimise_distribution`](@ref) runs that round trip in one call and hands
# back a `ToyDelay`, once `Bijectors` and an optimiser package are loaded.
# The optimiser itself stays the caller's choice.

using Bijectors, Optim

map_fit = optimise_distribution(delay, data, LBFGS())
map_fit.shape

# The fit starts at the template's own parameter values rather than at an
# arbitrary zero; `init` starts it somewhere else, and `loglik` swaps the
# reducer as it does for [`distribution_to_logdensity`](@ref).
#
# The steps are still individually available.
# [`logdensity_to_objective`](@ref) gives the callable,
# [`minimise`](@ref DistributionsInference.minimise) runs the optimiser, and
# [`objective_to_distribution`](@ref) maps the minimiser back onto the
# distribution.

f = logdensity_to_objective(prob)
z0 = DistributionsInference.to_unconstrained(prob, [delay.shape])
objective_to_distribution(prob, DistributionsInference.minimise(
    f, z0, LBFGS())).shape

# That is the maximum-a-posteriori point, because `logdensity` always adds an
# estimated row's own prior.
# Maximum likelihood needs a prior flat enough to leave the likelihood alone.

struct ToyDelayML{T <: Real}
    shape::T
    scale::T
end

function Distributions.logpdf(d::ToyDelayML, y::Real)
    return logpdf(Weibull(d.shape, d.scale), y)
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
optimise_distribution(ml_delay, data, LBFGS()).shape

# The data pull the shape above the prior mean of 2 either way, and the diffuse
# prior pulls it back less.

# ## Next
#
# - [Sampling with Turing](@ref turing-sampling) fits this same distribution
#   through `DynamicPPL` and reads the chain back with the same calls.
# - [Fitting a composed distribution](@ref composed-distributions) runs both
#   routes against a tree built by `ComposedDistributions`. Its extension
#   writes the protocol methods.
# - [Public API](@ref public-api) lists the rest of the protocol
#   ([`with_priors`](@ref),
#   [`estimated_rows`](@ref DistributionsInference.estimated_rows),
#   [`extra_logprior`](@ref DistributionsInference.extra_logprior)).
