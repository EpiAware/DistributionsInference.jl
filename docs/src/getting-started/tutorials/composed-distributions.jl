# # [Fitting a composed distribution](@id composed-distributions)
#
# A tree built with
# [ComposedDistributions](https://composeddistributions.epiaware.org/dev/) is
# fittable as it stands.
# Loading both packages activates an extension that maps the tree's own
# `params_table` onto this package's row schema, so every verb from
# [Fitting a custom distribution](@ref custom-distribution) works on the tree.
#
# This tutorial fits a two-event pathway through both routes, then fits a
# partially pooled tree.

using DistributionsInference, Distributions, Random
using ComposedDistributions
using ComposedDistributions: compose, uncertain, pool, event
using FlexiChains: FlexiChain, Parameter

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))

# ## What the tree declares
#
# [`parameter_rows`](@ref) reports every parameter in the tree, keyed by
# `edge.parameter`.
# The `onset_admit` shape carries the prior attached by `uncertain`, so it is
# estimated; everything else has no prior and stays at its value.

parameter_rows(tree)

# One row of the four carries a prior, so the fit has one parameter.

DistributionsInference.flat_dimension(tree)

# ## Fitting it
#
# A tree simulates the records it scores, so the data here are 200 draws from
# the tree itself, each a named delay per event.
# A fit should land near the shape those draws were generated at, 2.

rng = Xoshiro(1)
tree_data = [rand(rng, tree) for _ in 1:200]
tree_data[1]

# From here the calls are the ones the hand-written distribution used, pointed
# at `tree`.

prob = distribution_to_logdensity(tree, tree_data)

using AdvancedMH
using LinearAlgebra: I

model = AdvancedMH.DensityModel() do x
    any(<=(0), x) ? -Inf : DistributionsInference.logdensity(prob, x)
end
sampler = RWMH(MvNormal(zeros(1), 0.05^2 * I))
transitions = sample(Xoshiro(1), model, sampler, 2000;
    param_names = ["onset_admit.shape"], progress = false)
draws = [t.params for t in transitions][1001:end]

# The readback reads a `FlexiChain` keyed by the estimated rows' dotted names.
# Building that chain out of raw draws is FlexiChains' own constructor rather
# than a step this package owns, and a tree names its own rows, so key the
# draws by whatever `estimated_rows` declares instead of by hand.

function as_chain(obj, raw_draws)
    rows = DistributionsInference.estimated_rows(obj)
    values = permutedims(stack(raw_draws))
    return FlexiChain{Symbol}(size(values, 1), 1,
        Dict(Parameter(row.name) => reshape(values[:, i], :, 1)
        for (i, row) in enumerate(rows)))
end

fitted = point_estimate(tree, as_chain(tree, draws))

# The fit comes back as a tree, so its nodes are reachable by name.

event(fitted, :onset_admit)

# [`distribution_to_turing`](@ref) builds the same model over a tree, one site
# per estimated row.

using DynamicPPL, Turing
using FlexiChains: VNChain

Random.seed!(1)
turing_chain = sample(distribution_to_turing(tree, tree_data), NUTS(), 500;
    chain_type = VNChain, progress = false)
event(point_estimate(tree, turing_chain), :onset_admit)

# ## Partial pooling
#
# `pool` ties a parameter across branches through a shared population
# distribution, so three districts share what they know about their delay
# without being forced to agree.

population = uncertain(LogNormal(log(2.0), 0.3);
    mu = Normal(log(2.0), 0.2),
    sigma = truncated(Normal(0.0, 0.3); lower = 0.0))
pooled = compose((
    north = uncertain(Gamma(2.0, 1.0); shape = pool(:district, population)),
    east = uncertain(Gamma(2.0, 1.0); shape = pool(:district, population)),
    south = uncertain(Gamma(2.0, 1.0); shape = pool(:district, population))))

[row.name for row in DistributionsInference.estimated_rows(pooled)]

# A location-scale population is reparameterised non-centred, so what is
# estimated is the population's two hyperparameters and one standard normal
# offset per district.
# The estimation boundary moved and the fitting code is unchanged.

pooled_data = [rand(rng, pooled) for _ in 1:200]
Random.seed!(1)
pooled_chain = sample(distribution_to_turing(pooled, pooled_data), NUTS(0.9),
    500; chain_type = VNChain, progress = false,
    initial_params = InitFromPrior())
distribution_params(pooled, pooled_chain)

# The readback puts the offsets back through the population, so a district's
# shape comes out rather than its offset.

event(point_estimate(pooled, pooled_chain), :north)

# `InitFromPrior` above is Turing's own.
# The default initialisation draws uniformly on the unconstrained scale, which
# for a hierarchical shape can start the chain at `exp(16)` and underflow the
# likelihood.

# ## The one tree Turing refuses
#
# A centred pool scores its members against the reconstructed population rather
# than against a fixed prior of their own, and `DynamicPPL` has no sampling
# path for that yet.
# `distribution_to_turing` says so instead of mis-scoring the model.

centred_pool() = pool(:region, LogNormal(log(2.0), 0.3); noncentred = false)
centred = compose((
    north = uncertain(Gamma(2.0, 1.0); shape = centred_pool()),
    south = uncertain(Gamma(2.0, 1.0); shape = centred_pool())))

centred_data = [rand(rng, centred) for _ in 1:200]

try
    distribution_to_turing(centred, centred_data)
catch err
    println(sprint(showerror, err))
end

# The log-density route has no such gap, so a centred tree fits through
# [`distribution_to_logdensity`](@ref) and a gradient-free sampler.

centred_prob = distribution_to_logdensity(centred, centred_data)
centred_model = AdvancedMH.DensityModel() do x
    any(<=(0), x) ? -Inf : DistributionsInference.logdensity(centred_prob, x)
end
centred_sampler = RWMH(MvNormal(zeros(2), 0.05^2 * I))
centred_transitions = sample(Xoshiro(1), centred_model, centred_sampler, 2000;
    param_names = ["north.shape", "south.shape"], progress = false)
centred_draws = [t.params for t in centred_transitions][1001:end]
event(point_estimate(centred, as_chain(centred, centred_draws)), :north)

>>>>>>> 95fa41b (refactor!: take the FlexiChains conversion off the public surface)
# ## Next
#
# - ComposedDistributions' [verb map](https://composeddistributions.epiaware.org/dev/getting-started/concepts)
#   covers `compose`, `uncertain`, `pool` and `update`, the verbs that built
#   the trees here.
# - [Public API](@ref public-api) lists the protocol this extension implements
#   on a tree's behalf.
