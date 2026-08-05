# The one-call optimiser fit: `optimise_distribution` composed over
# `to_unconstrained`, `logdensity_to_objective`, `minimise` and
# `objective_to_distribution` (#106). Both fixtures here have a closed-form
# optimum, so the fit is checked against the answer rather than against the
# optimiser agreeing with itself.

@testsnippet OptimiseFixture begin
    using DistributionsInference, Distributions

    # An identity-linked parameter: a Normal mean under a Normal prior, with a
    # fixed known scale. The unconstrained scale is the constrained one (the
    # log-Jacobian is zero), so the optimum is the posterior mean.
    struct NormalMeanLeaf{M <: Real}
        mu::M
        sigma::Float64
        mu_prior::Distribution
    end

    function Distributions.logpdf(d::NormalMeanLeaf, y::Real)
        return logpdf(Normal(d.mu, d.sigma), y)
    end

    function DistributionsInference.parameter_rows(d::NormalMeanLeaf)
        return [
            (name = :mu, value = d.mu, prior = d.mu_prior,
                support = (-Inf, Inf)),
            (name = :sigma, value = d.sigma, prior = nothing,
                support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(
            d::NormalMeanLeaf, x::AbstractVector)
        return NormalMeanLeaf(x[1], d.sigma, d.mu_prior)
    end

    # The posterior mean of `mu`, the point the fit must find.
    function normal_mean_map(leaf, data)
        prior = leaf.mu_prior
        precision = 1 / var(prior) + length(data) / leaf.sigma^2
        weighted = mean(prior) / var(prior) + sum(data) / leaf.sigma^2
        return weighted / precision
    end

    # A log-linked parameter: an exponential rate under a Gamma prior. The
    # objective carries the log link's Jacobian, so its optimum is
    # `(a + n) / (1 / b + sum(data))` in closed form.
    struct RateLeaf{R <: Real}
        rate::R
        rate_prior::Distribution
    end

    function Distributions.logpdf(d::RateLeaf, y::Real)
        return logpdf(Exponential(1 / d.rate), y)
    end

    function DistributionsInference.parameter_rows(d::RateLeaf)
        return [(name = :rate, value = d.rate, prior = d.rate_prior,
            support = (0.0, Inf))]
    end

    function DistributionsInference.reconstruct(
            d::RateLeaf, x::AbstractVector)
        return RateLeaf(x[1], d.rate_prior)
    end

    function rate_map(leaf, data)
        a, b = params(leaf.rate_prior)
        return (a + length(data)) / (1 / b + sum(data))
    end
end

@testitem "the Optim extension loads" begin
    using DistributionsInference
    using Optim

    @test Base.get_extension(
        DistributionsInference, :DistributionsInferenceOptimExt) !== nothing
end

@testitem "minimise: returns the minimising vector" begin
    using DistributionsInference
    using Optim

    z_hat = DistributionsInference.minimise(
        z -> sum(abs2, z .- [2.0, -1.0]), [0.0, 0.0], LBFGS())
    @test z_hat≈[2.0, -1.0] atol=1e-6
end

@testitem "minimise: names Optim for an optimiser it has no method for" begin
    using DistributionsInference
    using Optim

    # `Optim` is loaded here, so this is the caller passing something that is
    # not an optimiser rather than a missing extension.
    thrown = try
        DistributionsInference.minimise(sum, [0.0], "LBFGS")
        nothing
    catch caught
        caught
    end
    @test thrown isa ArgumentError
    @test occursin("no method for an optimiser of type String", thrown.msg)
    @test occursin("Optim", thrown.msg)
end

@testitem "optimise_distribution: recovers an identity-linked optimum" setup=[OptimiseFixture] begin
    using Bijectors, Optim

    leaf = NormalMeanLeaf(0.0, 1.5, Normal(1.0, 2.0))
    data = [2.4, 1.1, 3.6, 0.8, 2.9, 1.7]
    fitted = optimise_distribution(leaf, data, LBFGS())

    @test fitted isa NormalMeanLeaf
    @test fitted.mu≈normal_mean_map(leaf, data) atol=1e-6
    # The fixed row is untouched and the template is unchanged.
    @test fitted.sigma == leaf.sigma
    @test leaf.mu == 0.0
end

@testitem "optimise_distribution: recovers a log-linked optimum" setup=[OptimiseFixture] begin
    using Bijectors, Optim

    leaf = RateLeaf(1.0, Gamma(2.0, 0.5))
    data = [0.4, 1.2, 0.7, 2.1, 0.9, 1.5, 0.3]
    fitted = optimise_distribution(leaf, data, LBFGS())

    @test fitted isa RateLeaf
    @test fitted.rate≈rate_map(leaf, data) atol=1e-6
    @test fitted.rate > 0
end

@testitem "optimise_distribution: the same optimum from a named start" setup=[OptimiseFixture] begin
    using Bijectors, Optim

    leaf = RateLeaf(1.0, Gamma(2.0, 0.5))
    data = [0.4, 1.2, 0.7, 2.1, 0.9, 1.5, 0.3]

    # Starting far from the template's own value finds the same point, so the
    # default start is a convenience rather than part of the answer.
    from_far = optimise_distribution(leaf, data, LBFGS(); init = [3.0])
    @test from_far.rate≈rate_map(leaf, data) atol=1e-6

    # And the default start is the template's values on the unconstrained
    # scale, not an arbitrary zero.
    prob = distribution_to_logdensity(leaf, data)
    @test DistributionsInference.to_unconstrained(prob, [leaf.rate]) ≈
          [log(leaf.rate)]
end

@testitem "optimise_distribution: honours loglik and a prepared problem" setup=[OptimiseFixture] begin
    using Bijectors, Optim

    # A reducer scoring the same likelihood twice moves the optimum, which a
    # wrapper ignoring `loglik` could not reproduce.
    leaf = RateLeaf(1.0, Gamma(2.0, 0.5))
    data = [0.4, 1.2, 0.7, 2.1, 0.9, 1.5, 0.3]
    doubled(obj, records) = 2 * sum(y -> logpdf(obj, y), records)

    fitted = optimise_distribution(leaf, data, LBFGS(); loglik = doubled)
    a, b = params(leaf.rate_prior)
    expected = (a + 2 * length(data)) / (1 / b + 2 * sum(data))
    @test fitted.rate≈expected atol=1e-6

    # The same fit from a `FitLogDensity` already to hand.
    prob = distribution_to_logdensity(leaf, data; loglik = doubled)
    @test optimise_distribution(prob, LBFGS()).rate ≈ fitted.rate
end

@testitem "optimise_distribution: an object estimating nothing comes back as it went in" setup=[ToyFixture] begin
    using Optim

    # No estimated row means nothing to optimise, so this needs neither an
    # optimiser call nor `Bijectors`.
    fixed_leaf = ToyGammaLeaf(2.0, 1.0)
    @test DistributionsInference.flat_dimension(fixed_leaf) == 0
    @test optimise_distribution(fixed_leaf, [1.5, 2.0], LBFGS()) == fixed_leaf
end

@testitem "optimise_distribution: the lower-level pieces reach the same point" setup=[OptimiseFixture] begin
    using Bijectors, Optim

    # The hand-wired route the wrapper replaces still works and agrees with
    # it, so nothing is lost by taking the one-call form.
    leaf = RateLeaf(1.0, Gamma(2.0, 0.5))
    data = [0.4, 1.2, 0.7, 2.1, 0.9, 1.5, 0.3]
    prob = distribution_to_logdensity(leaf, data)

    z0 = DistributionsInference.to_unconstrained(prob, [leaf.rate])
    z_hat = DistributionsInference.minimise(
        logdensity_to_objective(prob), z0, LBFGS())
    by_hand = objective_to_distribution(prob, z_hat)

    @test by_hand.rate ≈ optimise_distribution(leaf, data, LBFGS()).rate
end
