# # [Sampling with Turing](@id turing-sampling)
#
# The log-density is PPL-neutral, so Turing is a layer on top of it.
# Loading `DynamicPPL` activates [`distribution_to_turing`](@ref), which wraps
# the same problem as a `DynamicPPL` model: each estimated row becomes a named
# site drawn from its own prior, and the data likelihood is added from the
# distribution rebuilt at the draw.
#
# This tutorial fits the delay distribution from
# [Fitting a custom distribution](@ref custom-distribution) with `NUTS`, reads
# the chain back with the same two verbs, and swaps in a different likelihood.
#
# The distribution and its two protocol methods are repeated so this page runs
# on its own.

using DistributionsInference, Distributions, Random
using DynamicPPL, Turing
using FlexiChains: VNChain

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

# ## The model
#
# `distribution_to_turing` takes the distribution and its data, exactly as
# `distribution_to_logdensity` does.
# Its sites are the estimated rows, named as the protocol named them under the
# model's prefix (`d` by default, set with the `prefix` keyword).
# The fixed `scale` row has no prior, so it is not a site.

model = distribution_to_turing(delay, data)
keys(VarInfo(model))

# It samples like any other Turing model.
# `NUTS` differentiates through `reconstruct`, which is why the fields of
# `ToyDelay` are typed loosely enough to carry a dual number.

Random.seed!(1)
chain = sample(model, NUTS(), 500; chain_type = VNChain, progress = false)

# ## Reading the chain back
#
# The sites are keyed by the same dotted names the protocol declared, under the
# model's prefix, so [`point_estimate`](@ref) and [`readback_draws`](@ref) read
# a `VNChain` exactly as they read a hand-rolled chain.

point_estimate(delay, chain).shape

# Keeping every draw gives a posterior-predictive summary.

fits = readback_draws(delay, chain)
quantile([mean(Weibull(d.shape, d.scale)) for d in fits], [0.025, 0.5, 0.975])

# The readback contract is the dotted names, so switching sampler leaves this
# code alone.

# ## A different likelihood
#
# `distribution_to_turing` takes the same `loglik` reducer as
# `distribution_to_logdensity`.
# Aggregated records, a delay and the number of cases reporting it, score
# through a weighted sum rather than one term per row, and `NUTS` samples the
# result with no other change.

weighted_loglik(obj, records) = sum(w * logpdf(obj, y) for (y, w) in records)
counts = [(1.5, 12.0), (2.0, 30.0), (3.2, 8.0), (1.8, 21.0), (2.6, 15.0)]
Random.seed!(1)
weighted_chain = sample(
    distribution_to_turing(delay, counts; loglik = weighted_loglik),
    NUTS(), 500; chain_type = VNChain, progress = false)
point_estimate(delay, weighted_chain).shape

# The survival reducer from
# [Fitting a custom distribution](@ref custom-distribution) goes through the
# same keyword: `logccdf` for a `Weibull` differentiates, so `NUTS` samples
# that one unchanged too.

# ## Next
#
# - [Fitting a composed distribution](@ref composed-distributions) runs this
#   same model builder over a `ComposedDistributions` tree.
# - [Public API](@ref public-api) lists the rest of the protocol.
