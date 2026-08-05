# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# The docs navigation tree, read by `make.jl`. Every entry must resolve to a
# page the package writes or `make.jl` generates. Each tutorial leaf is the
# `.md` Literate renders from the `.jl` source registered in `docs_config.jl`.

const _TUTORIALS = "getting-started/tutorials"

pages = [
    "Home" => "index.md",
    "Getting started" => [
        "Overview" => "getting-started/index.md",
        "Tutorials" => [
            "Fitting a custom distribution" => "$_TUTORIALS/custom-distribution.md",
            "Sampling with Turing" => "$_TUTORIALS/turing.md",
            "Fitting a composed distribution" => "$_TUTORIALS/composed-distributions.md"
        ]
    ],
    "API reference" => [
        "Public API" => "lib/public.md",
        "Internal API" => "lib/internals.md"
    ]
]
