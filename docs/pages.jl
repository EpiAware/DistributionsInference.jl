# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# The docs navigation tree, read by `make.jl`. Every entry must resolve to a
# page the package writes or `make.jl` generates.

pages = [
    "Home" => "index.md",
    "Getting started" => [
        "Overview" => "getting-started/index.md",
        "Fitting an object" => "getting-started/inference.md",
        "Tutorials" => [
            "Automatic differentiation backends" => "getting-started/tutorials/ad-backends.md"
        ]
    ],
    "API reference" => [
        "Public API" => "lib/public.md",
        "Internal API" => "lib/internals.md"
    ]
]
