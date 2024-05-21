Project.toml is manually created
Manifest.toml is generated automatically from
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()


something similar in compiler


julia --startup-file=no --project=compiler compiler/build.jl foo
