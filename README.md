# DistributionsInference <img src="docs/src/assets/logo.svg" width="150" alt="DistributionsInference logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://distributionsinference.epiaware.org/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://distributionsinference.epiaware.org/dev/) | [![Test](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg)](https://codecov.io/gh/EpiAware/DistributionsInference.jl) [![AD](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/EpiAware/DistributionsInference.jl/actions/workflows/ad.yaml) | [![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FDistributionsInference&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/DistributionsInference) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FDistributionsInference&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/DistributionsInference) |

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-reversediff) | [![cov Enzyme forward](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/EpiAware/DistributionsInference.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/EpiAware/DistributionsInference.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

The inference layer for the EpiAware composable-modelling stack.
A distribution names its own parameters and becomes fittable through a PPL-neutral log-density, with no commitment to a probabilistic programming language.

## Why DistributionsInference?

- Fitting a distribution usually means rewriting it inside one probabilistic programming language's macros; here a distribution names its own scalar parameters and is fitted as it stands.
- `distribution_to_logdensity` turns a distribution and its data into a `LogDensityProblems` problem, so anything that works with `LogDensityProblems` works on the distribution.
- A distribution declares its parameters as a table of rows, one row per scalar parameter carrying its name, value, prior and support.
  Attaching a prior to a row is what makes that parameter estimated.
  The rows are plain `NamedTuple`s, so the inventory is a row table any Tables.jl consumer reads, without this package depending on Tables.jl.
- `point_estimate` post-processes sampler output onto a fitted distribution in one call.
  It is generic across samplers and extensible to a new chain type by adding a method.
- The distribution that comes back is the same kind of object that went in, so a fitted `Gamma` is a `Gamma` and can be used anywhere a distribution can.

## Getting started

See [documentation](https://distributionsinference.epiaware.org/dev/) for a full walkthrough.

Start with an ordinary `Distributions.jl` distribution, a `Gamma`, and no type of your own.
The package ships no `parameter_rows`/`reconstruct` methods for `Distributions.jl` types, so those two methods are yours to write, once per family.
That is a gap in the package rather than a step the design asks for; nothing else about the distribution changes.

```julia
using DistributionsInference, Distributions, Random

function DistributionsInference.parameter_rows(d::Gamma)
    return [(name = :shape, value = shape(d),
            prior = LogNormal(log(2.0), 0.5), support = (0.0, Inf)),
        (name = :scale, value = scale(d),
            prior = LogNormal(0.0, 0.5), support = (0.0, Inf))]
end

function DistributionsInference.reconstruct(::Gamma, x::AbstractVector)
    return Gamma(x[1], x[2])
end
```

`parameter_rows` is the estimation boundary, in that a row carrying a prior is estimated and a row with `prior = nothing` stays fixed at its value.
`distribution_to_logdensity` packages a template distribution and the data into a log-density over the estimated rows.

```julia
template = Gamma(2.0, 1.0)
data = rand(Xoshiro(1), Gamma(2.0, 1.0), 200)
prob = distribution_to_logdensity(template, data)
DistributionsInference.flat_dimension(template)
```

`prob` is a `LogDensityProblems` problem, so any sampler that consumes that interface can drive it.
Here that is `AdvancedMH`'s random-walk Metropolis, wrapped in a guard because a random-walk proposal does not respect positive support on its own.

```julia
using AdvancedMH
using LinearAlgebra: I

model = AdvancedMH.DensityModel() do x
    any(<=(0), x) ? -Inf : DistributionsInference.logdensity(prob, x)
end
sampler = RWMH(MvNormal(zeros(2), 0.05^2 * I))
transitions = sample(Xoshiro(1), model, sampler, 4000;
    param_names = ["shape", "scale"], progress = false)
draws = [t.params for t in transitions][2001:end]
```

`point_estimate` reads the draws back onto the distribution, through the same dotted-name chain a real PPL's sampler would hand back.
What it returns is a `Gamma`, printed here rather than picked apart.
The chain readback is a package extension, so add and load `FlexiChains` for this last step; everything above needs only `DistributionsInference`.

```julia
using FlexiChains: FlexiChains

chain = to_flexichain(template, draws)
point_estimate(template, chain)
```

The [getting started guide](https://distributionsinference.epiaware.org/dev/getting-started/) goes further, with a distribution type of your own, every draw read back through `readback_draws`, and sampling with Turing instead of AdvancedMH.

## Related packages

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) defines the distributions this fits; a type that has a `logpdf` and can name its scalar parameters is a candidate.
- [ComposedDistributions.jl](https://composeddistributions.epiaware.org/dev/) builds a distribution by composing others into chains, branches and outcomes; a package extension here reads a composed tree's generated codec directly, so its estimated leaves, pooled and shared parameters included, are fittable with no extra glue.
- [ModifiedDistributions.jl](https://modifieddistributions.epiaware.org/dev/) wraps a distribution to change one behaviour, such as rescaling, likelihood weighting or a hazard shift; a modifier used as a leaf inside a composed tree is already fittable, but the extension for fitting a standalone modifier is parked until that package registers in General.
- [ReparameterisedDistributions.jl](https://reparameteriseddistributions.epiaware.org/dev/) switches a family between parameter conventions, so a distribution can be fitted in the coordinates its priors were elicited in rather than the family's native ones.
- [CensoredDistributions.jl](https://censoreddistributions.epiaware.org/dev/) applies primary event censoring, interval censoring and right truncation to a delay, and returns a `Distributions.jl` distribution.
- [ConvolvedDistributions.jl](https://convolveddistributions.epiaware.org/dev/) builds the distribution of a sum, difference or product of independent delays, again as a `Distributions.jl` distribution.

ComposedDistributions is the only one of these covered by an extension today.
For the rest, as for a plain `Distributions.jl` distribution, the two protocol methods are yours to write.

## Where to learn more

- Want to get started running code? See the [getting started guide](https://distributionsinference.epiaware.org/dev/getting-started/).
- Want to understand the API? See the [API reference](https://distributionsinference.epiaware.org/dev/lib/public).
- Want to see the code? Check out our [GitHub repository](https://github.com/EpiAware/DistributionsInference.jl).

## Getting help

For usage questions, ask on the [Julia Discourse](https://discourse.julialang.org)
(the SciML or usage categories) or the [epinowcast community forum](https://community.epinowcast.org),
our home for epidemiological modelling questions.
Please use [GitHub issues](https://github.com/EpiAware/DistributionsInference.jl/issues)
for bug reports and feature requests only.

<!-- standard-sections:start -->
<!-- MANAGED by EpiAwarePackageTools.scaffold — do not edit between the
     markers. These standard sections are re-rendered on every update;
     edit the package-owned sections outside them, or CITATION.cff. -->

## Part of the EpiAware ecosystem

DistributionsInference is part of [EpiAware](https://epiaware.org), a set of composable tools for infectious disease modelling. See the [other packages](https://github.com/EpiAware) in the ecosystem.

## Contributing

We welcome contributions and new contributors! Please open an issue or pull request on [GitHub](https://github.com/EpiAware/DistributionsInference.jl). This package follows [ColPrac](https://github.com/SciML/ColPrac) and the [SciML style](https://github.com/SciML/SciMLStyle).

## How to cite

If you use DistributionsInference in your work, please cite it. Citation metadata lives in [`CITATION.cff`](https://github.com/EpiAware/DistributionsInference.jl/blob/main/CITATION.cff), which GitHub renders as a "Cite this repository" button on the repository page.

## Code of conduct

Please note that the DistributionsInference project is released with a [Contributor Code of Conduct](https://github.com/EpiAware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing, you agree to abide by its terms.
<!-- standard-sections:end -->
