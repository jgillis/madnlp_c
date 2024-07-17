cmake_minimum_required(VERSION 3.12)
project(MadNLP_C VERSION 0.1.0)  # Adjust the version as necessary


message(CHECK_START "Check Julia binary dir")
set(Julia_EXECUTABLE /usr/bin/julia)
set(JULIA_BIN_DIR "nd" CACHE STRING "Julia runtime binary dir default")
if (JULIA_BIN_DIR STREQUAL "nd")
  execute_process(
    COMMAND ${Julia_EXECUTABLE} --startup-file=no -e "print(dirname(Sys.BINDIR))"
    OUTPUT_VARIABLE JULIA_BIN_DIR_PARSED
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set(JULIA_BIN_DIR ${JULIA_BIN_DIR_PARSED} CACHE STRING "Julia runtime binary dir" FORCE)
endif()
message(CHECK_PASS ${JULIA_BIN_DIR})

message(CHECK_START "Check MadNLP.jl version")
set(MADNLP_VERSION "nd" CACHE STRING "MadNLP Julia Library Version")
if (MADNLP_VERSION STREQUAL "nd")
  execute_process(
    COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=./ -e "using Pkg; Pkg.instantiate();m=Pkg.dependencies();v=m[findfirst(v->v.name==\"MadNLP\",m)].version; print(v);"
    OUTPUT_VARIABLE MADNLP_VERSION_PARSED
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set(MADNLP_VERSION ${MADNLP_VERSION_PARSED} CACHE STRING "MadNLP Julia Library Version" FORCE)
endif()
message(CHECK_PASS ${MADNLP_VERSION})

set(MADNLP_C_FULLNAME madnlp_c-${MADNLP_VERSION}-${PROJECT_VERSION})
set(BUILD_SCRIPT ${PROJECT_SOURCE_DIR}/compiler/build.jl)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/MadNLP_env_status.txt
  DEPENDS ${PROJECT_SOURCE_DIR}/Manifest.toml
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=./ -e "using Pkg; Pkg.instantiate()"
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=./ -e "using Pkg; Pkg.status()" > ${PROJECT_BINARY_DIR}/MadNLP_env_status.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Instantiate Julia packages"
  VERBATIM
)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/PackageCompiler_env_status.txt
  DEPENDS ${PROJECT_BINARY_DIR}/MadNLP_env_status.txt
  DEPENDS ${PROJECT_SOURCE_DIR}/compiler/Manifest.toml
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=compiler -e "using Pkg; Pkg.instantiate()"
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=compiler -e "using Pkg; Pkg.status()" > ${PROJECT_BINARY_DIR}/PackageCompiler_env_status.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Instantiate Julia build packages"
  VERBATIM
)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/library_precompilation_dummy.txt
  DEPENDS ${PROJECT_BINARY_DIR}/PackageCompiler_env_status.txt
  DEPENDS ${PROJECT_SOURCE_DIR}/src/MadNLP_C.jl
  DEPENDS ${PROJECT_SOURCE_DIR}/compiler/build.jl
  DEPENDS ${PROJECT_SOURCE_DIR}/compiler/generate_precompile.jl
  COMMAND echo 1 > ${PROJECT_BINARY_DIR}/library_precompilation_dummy.txt
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=compiler ${BUILD_SCRIPT} ${PROJECT_BINARY_DIR}/${MADNLP_C_FULLNAME}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Create Madnlp precompiled library sysimage"
  VERBATIM
)

add_custom_target(madnlp_c ALL
  DEPENDS ${PROJECT_BINARY_DIR}/library_precompilation_dummy.txt
)
