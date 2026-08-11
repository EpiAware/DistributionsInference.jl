# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Prepended to the GitHub releases fetched at build time to build the
# release notes page.

const RELEASE_NOTES_HEADER = """
```@meta
EditURL = "https://github.com/EpiAware/DistributionsInference.jl/releases"
```

# Release notes

Release notes live in [GitHub Releases](https://github.com/EpiAware/DistributionsInference.jl/releases), generated from the pull requests merged into each release.
The most recent are reproduced below, as they were written there.

"""
