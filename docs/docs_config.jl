# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Package-specific configuration read by the managed `make.jl`.

# Literate tutorial sources under `TUTORIALS_SUBDIR`. Light tutorials emit
# `@example` blocks Documenter runs in-process; heavy ones (live MCMC fits,
# multi-backend AD, plotting) each run in a fresh subprocess.
#
# The three fitting tutorials are light despite sampling: each fits at most
# five parameters from a handful of records, so they run in seconds and a fast
# build (`--skip-notebooks`) still executes them, which is where their code is
# checked. They need no `TUTORIAL_STUBS` entry for the same reason.
const LIGHT_TUTORIALS = String[
    "custom-distribution.jl",
    "turing.jl",
    "composed-distributions.jl"
]

# `ad-backends.jl` is kit-managed and re-applied on every sync, but it is not
# registered here, so the build neither runs it nor writes a page for it. It
# cannot run: `AlgebraOfGraphics` caps `DimensionalData` below the floor the
# `FlexiChains` these docs load needs (kit#283, and CD#147 for the same
# conflict), so its plotting stack is out of docs/Project.toml (#19); and the
# `ADFixtures` registry no longer imports Mooncake (DI#73), so its two
# `AutoMooncake` rows would fail even with the plotting stack back.
#
# Registering it and force-stubbing it instead would publish a page whose only
# body is a fast-build notice aimed at maintainers (kit#382). Register it here
# and in `pages.jl` once it can run.
const HEAVY_TUTORIALS = String[]

# Relative to `docs/src`.
const TUTORIALS_SUBDIR = joinpath("getting-started", "tutorials")

# Fast-build stubs (`--skip-notebooks`). Keep each tutorial's `@id` in the
# heading so cross-references from other pages still resolve.
const TUTORIAL_STUBS = Pair{String, String}[]

# Heavy tutorials that always render from their `TUTORIAL_STUBS` heading and
# never execute.
const FORCE_STUB_TUTORIALS = String[]

# Whether the README block and docs footer carry EpiAware ecosystem branding.
# The content this turns on is kit-managed; only this line is package-owned.
const ORG_BRANDING = true

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
#
# "Getting help" is dropped here so the docs carry it once, on the
# getting-started overview; the README keeps it for readers on GitHub.
const INDEX_STRIP_SECTIONS = String["Getting help"]

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
