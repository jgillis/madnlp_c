cmake_minimum_required(VERSION 3.12)
project(MadNLP_C VERSION 1.0.0)  # Adjust the version as necessary

set(Julia_EXECUTABLE julia)

add_custom_command(
  OUTPUT ${OUT_DIR}/build_step_stamp
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=${CMAKE_CURRENT_SOURCE_DIR} -e "using Pkg; Pkg.instantiate()"
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=build -e "using Pkg; Pkg.instantiate()"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Instantiating Julia packages"
)
