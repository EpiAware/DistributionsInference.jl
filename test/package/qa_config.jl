# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# QA configuration values the managed `quality.jl` testset reads.

using DistributionsInference

const QA_CONFIG = (
    # The module under test.
    mod = DistributionsInference,

    # Path to the isolated JET environment (see test/jet/Project.toml).
    jet_env = joinpath(@__DIR__, "..", "jet"),

    # Path to the isolated formatter environment. Runs the check in a
    # subprocess pinned to an exact JuliaFormatter version, rather than
    # whatever the shared test environment resolves (kit #321).
    formatter_env = joinpath(@__DIR__, "..", "formatter"),

    # Per-check Aqua relaxations, e.g. (; ambiguities = false). Empty = all on.
    aqua = (;),

    # ExplicitImports `ignore`: symbols the main module legitimately imports
    # non-publicly, e.g. (:_internal_helper,). Package extensions are handled
    # automatically (#189).
    ei_ignore = (),

    # Docstring `crossref_ignore`: upstream names docstrings link to via
    # `[`name`](@ref)`, e.g. (:pdf, :cdf, :logpdf).
    crossref_ignore = (),

    # Extra docstring-format options, e.g.
    # (; exported_only_examples = true, require_field_docs = true).
    docstring = (;),

    # README section-structure check. `path` is the package root. Override
    # `required`/`order` to extend or relax the standard section set, e.g.
    #   (; required = vcat(STANDARD_README_SECTIONS, [("Benchmarks",)]))
    readme = (; path = joinpath(@__DIR__, "..", "..")),

    # Package extensions to ambiguity-check. Each entry:
    #   (; name = :MyPkgSomeTriggerExt,
    #      triggers = ("SomeTrigger",),       # packages to load first
    #      prefixes = ("MyPkg", "SomeTrigger"),
    #      expect_phantoms = false,    # true if a third party adds phantoms
    #      broken = false)             # true to quarantine a known ambiguity
    #
    # Every extension whose triggers are test deps is listed; Aqua only sees
    # the core module's own methods. Mooncake is not a test dep, so its
    # extension cannot be loaded here.
    extensions = (
        (; name = :DistributionsInferenceFlexiChainsExt,
            triggers = ("FlexiChains",),
            prefixes = ("DistributionsInference", "FlexiChains")),
        (; name = :DistributionsInferenceDynamicPPLFlexiChainsExt,
            triggers = ("DynamicPPL", "FlexiChains"),
            prefixes = ("DistributionsInference", "DynamicPPL",
                "FlexiChains")),
        (; name = :DistributionsInferenceDynamicPPLExt,
            triggers = ("DynamicPPL",),
            prefixes = ("DistributionsInference", "DynamicPPL")),
        (; name = :DistributionsInferenceBijectorsExt,
            triggers = ("Bijectors",),
            prefixes = ("DistributionsInference", "Bijectors")),
        (; name = :DistributionsInferenceComposedDistributionsExt,
            triggers = ("ComposedDistributions",),
            prefixes = ("DistributionsInference", "ComposedDistributions"))
    )
)
