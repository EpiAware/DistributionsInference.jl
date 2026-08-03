## 0.1.0

The first release.
It contains the fit protocol (`parameter_rows`, `estimated_rows`, `flat_dimension`, `reconstruct`), the PPL-neutral log-density engine built on it (`FitLogDensity`, `as_logdensity`, `logdensity`, with a direct `LogDensityProblems` implementation), default-prior assembly (`default_prior`, `distribution_priors`) and the dotted-name chain readback (`to_flexichain`, `distribution_params`, `readback`, `readback_draws`).

Chain, PPL, transform and composed-distribution support are weakdep extensions over the same protocol rather than requirements: `FlexiChains` (the dotted-name readback), `DynamicPPL` (`as_turing`), `DynamicPPL` x `FlexiChains` (the `VarName`-keyed readback), `Bijectors` (`to_constrained` and `as_optimisation_objective`), `ComposedDistributions` (the protocol over CD's own codec) and `Mooncake` (`xlogy`/`xlog1py` gradient rules).
A project can start from the bare log-density and add a chain type or a PPL later.

Breaking, relative to the pre-release package:

- `FlexiChains` is a weak dependency, so using the readback means `using FlexiChains` alongside this package.
  The four readback functions keep stubs in the core module that name the package and extension to load rather than raising a bare `MethodError`.
- `extra_logprior` takes a fourth argument, `state`, which is `extra_prior_state(obj)` computed once when `as_logdensity` assembles a `FitLogDensity` (#28).
  A type overriding `extra_logprior` updates to the four-argument signature; an override needing no state can ignore the trailing argument.

Fixed: a `reconstruct` method whose estimated field is concretely typed (e.g. `shape::Float64`) failed a gradient-based sampler with an opaque `MethodError` from inside the struct's constructor.
`logdensity` and the `DynamicPPL` extension's model now guard this with a named `ArgumentError` before `reconstruct` runs (#48).

The `ModifiedDistributions` extension is parked until that package registers in General, which will not accept a package naming an unregistered one in `[weakdeps]` (#17).
The extension, its tests and its dependency entries came out in a single commit placed last on the release branch, so a follow-up release un-parks it by reverting that commit.
