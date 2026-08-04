# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Package-specific configuration read by the managed `make.jl`.

# Literate tutorial sources under `TUTORIALS_SUBDIR`. Light tutorials emit
# `@example` blocks Documenter runs in-process; heavy ones (live MCMC fits,
# multi-backend AD, plotting) each run in a fresh subprocess.
const LIGHT_TUTORIALS = String[]

# The `ad-backends.jl` page itself is kit-managed and re-applied on every
# sync; only this registration is package-owned.
const HEAVY_TUTORIALS = String[
    "ad-backends.jl"
]

# Relative to `docs/src`.
const TUTORIALS_SUBDIR = joinpath("getting-started", "tutorials")

# Fast-build stubs (`--skip-notebooks`). Keep each tutorial's `@id` in the
# heading so cross-references from other pages still resolve.
const TUTORIAL_STUBS = Pair{String, String}[
    "ad-backends.md" => "# [Automatic differentiation backends](@id ad-backends)"
]

# Heavy tutorials that always render from their `TUTORIAL_STUBS` heading and
# never execute. `ad-backends.jl` is stubbed while its plotting stack cannot
# resolve: `AlgebraOfGraphics` caps `DimensionalData` below the floor the
# `FlexiChains` these docs load needs (kit#283, and CD#147 for the same
# conflict), so those deps are out of docs/Project.toml (#19).
# Un-stub and restore them when that ceiling lifts.
#
# The page also loads no AD backend of its own, and the `ADFixtures` registry
# no longer imports Mooncake (DI#73), so its two `AutoMooncake` rows will fail
# until the managed page gains a `using Mooncake` or the registry imports it
# again; the `Mooncake` docs dep installs it but does not load it.
const FORCE_STUB_TUTORIALS = String["ad-backends.jl"]

# Whether the README block and docs footer carry EpiAware ecosystem branding.
# The content this turns on is kit-managed; only this line is package-owned.
const ORG_BRANDING = false

# URLs to skip during the (full-build) linkcheck. The docs site has never
# deployed, so the subdomain has no TLS certificate and every self-link fails;
# discussions are not enabled on the repo.
const LINKCHECK_IGNORE = [
    r"^https://distributionsinference\.epiaware\.org",
    r"github\.com/EpiAware/DistributionsInference\.jl/discussions"
]

# README -> index.md link rewrites, `from => to`, applied line by line.
const INDEX_REWRITES = Pair{String, String}[]

# Whether README ```julia blocks become runnable `@example readme` blocks on
# the generated home page. Set `false` where they are illustrative.
const README_EXECUTE = true

# README headings whose whole section is dropped from the home page. The
# managed badge block is always stripped via its `<!-- badges:start -->` /
# `<!-- badges:end -->` markers.
const INDEX_STRIP_SECTIONS = String[]

# Whether the build generates the benchmark page (`src/benchmarks.md`); `false`
# also drops its `pages.jl` nav entry.
const BENCHMARK_PAGE = false

# Headline benchmark suites to keep on the performance-history page, named by
# the first `/`-segment of a benchmark's name. Empty keeps every suite.
const HISTORY_SUITES = String[]

# How many of the most-recent revisions (columns) to show in the summary and
# history tables.
const HISTORY_COMMITS = 5

# A suite flags "⚠ reg" when its median at the most recent shown revision,
# divided by its median at the oldest, exceeds this. Must be > 1.0, else an
# unchanged or improved suite flags.
const HISTORY_REGRESSION_THRESHOLD = 1.1
