cmake_minimum_required(VERSION 3.5)
project(madnlp_test)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_VERBOSE_MAKEFILE 1)
set(VERBOSE  1)

set(MADNLP_DIR ${PROJECT_SOURCE_DIR}/../build/madnlp_c-0.8.3-0.1.0)
set(MADNLP_INCLUDE_DIR ${MADNLP_DIR}/include)
set(MADNLP_LIB_DIR ${MADNLP_DIR}/lib/)

include(LocalInclude OPTIONAL)
add_compile_options(-Wall -ggdb -std=c++11)
add_executable(test_madnlp_c main.cpp)

target_include_directories(test_madnlp_c PUBLIC ${MADNLP_INCLUDE_DIR})
target_link_libraries(test_madnlp_c  -L${MADNLP_LIB_DIR} -lmadnlp_c -ljulia -lstdc++ -ldl -lm)
