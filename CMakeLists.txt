cmake_minimum_required(VERSION 3.12)
project(MadNLP_C VERSION 1.0.0)  # Adjust the version as necessary

set(Julia_EXECUTABLE julia)

add_custom_command(
  OUTPUT FOO
  COMMAND ${Julia_EXECUTABLE} --startup-file=no --project=compiler -e "using Pkg; Pkg.instantiate()"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Instantiating Julia packages"
)


