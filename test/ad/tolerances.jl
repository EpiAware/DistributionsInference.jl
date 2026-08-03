# PACKAGE-OWNED — the AD matrix's comparison tolerance.
#
# The shared harness defaults to `rtol = 5e-2`, sized for scenarios whose
# gradients come out of a sampler or a quadrature and carry real numerical
# noise. Every scenario in this package's registry is a smooth analytic scalar
# function of one or two parameters, where a correct backend agrees with the
# ForwardDiff reference to near machine precision, so 5% would wave through
# exactly the failure class the registry exists to catch: a
# `LogExpFunctions.xlogy`/`xlog1py` rule that is right in the limit and wrong
# in a low-order term (DI#7, ComposedDistributions#99).
#
# `atol` matters as much as `rtol` here: `isapprox` compares gradient VECTORS
# by norm, so on a two-parameter scenario a small coordinate can be wrong by
# far more than `rtol` of its own value while the larger coordinate keeps the
# norm inside tolerance. Both are tightened together.
#
# Tighten or loosen per backend rather than globally if one ever needs slack,
# and record which backend and why.
@testsnippet ADTolerances begin
    const AD_RTOL = 1e-8
    const AD_ATOL = 1e-10
end
