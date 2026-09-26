using Documenter
using SigmaClip

DocMeta.setdocmeta!(SigmaClip, :DocTestSetup, :(using SigmaClip); recursive = true)

makedocs(;
    modules = [SigmaClip],
    sitename = "SigmaClip.jl",
    authors = "Matteo Sapori",
    format = Documenter.HTML(;
        canonical = "https://m4ttes4.github.io/SigmaClip.jl",
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/m4ttes4/SigmaClip.jl", devbranch = "main", push_preview = false)
