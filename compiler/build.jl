import PackageCompiler, TOML

if length(ARGS) < 1 || length(ARGS) > 2
    println("Usage: julia $PROGRAM_FILE target_dir [major|minor]")
    println()
    println("where:")
    println("    target_dir is the directory to use to create the library bundle")
    println("    [major|minor] is the (optional) compatibility version (default: major).")
    println("                  Use 'minor' if you use new/non-backwards-compatible functionality.")
    println()
    println("[major|minor] is only useful on OSX.")
    exit(1)
end

const build_dir = @__DIR__
const src_dir = realpath(joinpath(build_dir, "..", "src"))
const target_dir = ARGS[1]
const project_toml = realpath(joinpath(build_dir, "..", "Project.toml"))
const version = VersionNumber(TOML.parsefile(project_toml)["version"])

const compatibility = length(ARGS) == 2 ? ARGS[2] : "major"

PackageCompiler.create_library(".", target_dir;
                            lib_name="madnlp_c",
                            precompile_execution_file=[joinpath(build_dir, "generate_precompile.jl")],
                            precompile_statements_file=[joinpath(build_dir, "additional_precompile.jl")],
                            incremental=false,
                            force=true,
                            filter_stdlibs=false,
                            include_lazy_artifacts=true,
                            header_files = [joinpath(src_dir, "madnlp_c.h")],
                            version=version,
                            compat_level=compatibility,
                        )
