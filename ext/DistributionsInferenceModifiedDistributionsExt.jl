# DistributionsInference × ModifiedDistributions: the fit protocol
# (`parameter_rows`/`reconstruct`) for a STANDALONE modifier distribution
# (`affine(Gamma(...))`, `thin(...)`, ...) — not as a leaf inside a composed
# tree, which DI#5's ComposedDistributions extension already covers
# transitively (that extension drives CD's codec, which peels a modifier leaf
# through CD's OWN leaf-protocol hooks in
# `ModifiedDistributionsComposedDistributionsExt`).
#
# This extension cannot reach that peel-through machinery: it weakdeps
# ModifiedDistributions alone (triggered by `using ModifiedDistributions`
# without ComposedDistributions), so it walks the modifier layers itself via
# ModifiedDistributions' own public `get_dist`/`get_op`/`get_factor` accessors,
# with its own native-parameter-name table (mirrors ComposedDistributions'
# `_param_names`, ported by hand — this extension cannot depend on CD, see
# ComposedDistributions#185's one-directional dependency edge).
#
# Every row this reports is FIXED (`prior = nothing`): a bare
# `Distributions.jl` leaf carries no prior anywhere, so there is nothing here
# to mark estimated on the object itself, and `flat_dimension` is always `0`.
# That is not a stand-in for a missing feature — it is the complete, correct
# scope. Following the pattern shipped in DistributionsInference#23,
# `distribution_priors(obj; priors, default)` (`src/priors.jl`) is the
# generic "attach a prior" path for ANY `parameter_rows`-implementing object:
# it returns a full row set with priors filled, directly usable as a fittable
# object via the bare-row-vector identity, paired with a caller-supplied
# `loglik` that rebuilds a concrete distribution from the row values (the
# exact pattern DI's own core test suite demonstrates for a bare `Gamma`).
# `parameter_rows` below is what makes that path available to a standalone
# modifier; no bespoke "make this Affine estimated" mechanism is needed.
module DistributionsInferenceModifiedDistributionsExt

using Distributions: Distributions
using ModifiedDistributions: ModifiedDistributions, AbstractModifiedDistribution,
                             Transformed, ThinOp, get_dist, get_op, get_factor
import DistributionsInference: parameter_rows, reconstruct

# The native parameter names of a peeled (non-modifier) leaf, matched
# positionally to `Distributions.params`. Mirrors ComposedDistributions'
# `_param_names` table exactly (same mapped families); an unmapped family
# falls back to positional `:param_i` names below.
_native_param_names(::Distributions.Normal) = (:mu, :sigma)
_native_param_names(::Distributions.LogNormal) = (:mu, :sigma)
_native_param_names(::Distributions.Gamma) = (:shape, :scale)
_native_param_names(::Distributions.Weibull) = (:shape, :scale)
_native_param_names(::Distributions.Exponential) = (:scale,)
_native_param_names(::Distributions.Uniform) = (:lower, :upper)
_native_param_names(::Any) = ()

# Walk down through nested modifier layers via `get_dist`, collecting each
# layer's own modifier-owned extra parameter row along the way (currently only
# a `Transformed` carrying a `ThinOp`'s thinning factor, mirroring
# ModifiedDistributionsComposedDistributionsExt's `extra_leaf_params`), until
# reaching the innermost non-modifier distribution. Returns `(native,
# extra_rows)`, `extra_rows` in outer-to-inner encounter order. Two nested
# `thin(...)` layers would both report a `:thin` row and collide by name — a
# real but contrived case (thinning twice is unusual modelling), not
# disambiguated here.
function _peel_and_collect_extras(d)
    extras = NamedTuple{(:name, :value, :prior, :support)}[]
    node = d
    while node isa AbstractModifiedDistribution
        if node isa Transformed
            op = get_op(node)
            if op isa ThinOp
                push!(extras,
                    (name = :thin, value = get_factor(op), prior = nothing,
                        support = (0.0, 1.0)))
            end
        end
        node = get_dist(node)
    end
    return node, extras
end

# `parameter_rows`: the peeled native leaf's own parameters (support the whole
# leaf's `(minimum, maximum)`, mirroring ComposedDistributions' leaf rows),
# then each layer's extra parameter, in encounter order. Every row is FIXED
# (`prior = nothing`) — see this file's header note.
function parameter_rows(d::AbstractModifiedDistribution)
    native, extras = _peel_and_collect_extras(d)
    vals = Distributions.params(native)
    base = _native_param_names(native)
    names = ntuple(length(vals)) do i
        i <= length(base) ? base[i] : Symbol(:param_, i)
    end
    sup = (Distributions.minimum(native), Distributions.maximum(native))
    native_rows = [(name = names[i], value = vals[i], prior = nothing, support = sup)
                   for i in eachindex(vals)]
    return vcat(native_rows, extras)
end

# `reconstruct`: every row is fixed (see `parameter_rows`), so
# `flat_dimension(d)` is always `0` and the only valid call is the identity
# round-trip at the empty vector, mirroring the fixed-object case of every
# other fit-protocol implementation in this repository (e.g. `ToyGammaLeaf`).
function reconstruct(d::AbstractModifiedDistribution, x::AbstractVector)
    isempty(x) || throw(DimensionMismatch(
        "$(typeof(d)) has 0 estimated parameters (parameter_rows reports " *
        "every row fixed); got a flat vector of length $(length(x))"))
    return d
end

end # module DistributionsInferenceModifiedDistributionsExt
