cmake_minimum_required(VERSION 3.12)
project(MadNLP_C VERSION 1.0.0)  # Adjust the version as necessary

set(Julia_EXECUTABLE /usr/bin/julia)
set(BUILD_SCRIPT ${PROJECT_SOURCE_DIR}/compiler/build.jl)
set(MADNLP_VERSION "nd" CACHE STRING "MadNLP Julia Library Version")

message(CHECK_START "Reading MadNLP.jl version")
execute_process(
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=./ -e "using Pkg; Pkg.instantiate();m=Pkg.dependencies();v=m[findfirst(v->v.name==\"MadNLP\",m)].version; print(v);"
  OUTPUT_VARIABLE MADNLP_VERSION
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
message(CHECK_PASS ${MADNLP_VERSION})

set(MADNLP_C_FULLNAME madnlp_c-${MADNLP_VERSION}-${PROJECT_VERSION})

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/MadNLP_env_status.txt
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=./ -e "using Pkg; Pkg.instantiate()"
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=./ -e "using Pkg; Pkg.status()" > ${PROJECT_BINARY_DIR}/MadNLP_env_status.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Instantiating Julia packages"
  VERBATIM
)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/PackageCompiler_env_status.txt
  DEPENDS ${PROJECT_BINARY_DIR}/MadNLP_env_status.txt
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=./ -e "using Pkg; Pkg.instantiate()"
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=compiler -e "using Pkg; Pkg.instantiate()"
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=compiler -e "using Pkg; Pkg.status()" > ${PROJECT_BINARY_DIR}/PackageCompiler_env_status.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Instantiating Julia build packages"
  VERBATIM
)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/${MADNLP_C_FULLNAME}
  DEPENDS ${PROJECT_BINARY_DIR}/PackageCompiler_env_status.txt
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=compiler ${BUILD_SCRIPT} ${PROJECT_BINARY_DIR}/${MADNLP_FULLNAME}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Create Madnlp precompiled library sysimage"
)

add_custom_target(madnlp_c ALL
  DEPENDS ${PROJECT_BINARY_DIR}/${MADNLP_C_FULLNAME}
)
