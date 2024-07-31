Project.toml is manually created
Manifest.toml is generated automatically from

```
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=compiler -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=compiler compiler/build.jl target/
```
