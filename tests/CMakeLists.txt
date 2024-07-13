cmake_minimum_required(VERSION 3.5)
project(madnlp_test)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_VERBOSE_MAKEFILE 1)
set(VERBOSE  1)

set(CASADI_DIR ${PROJECT_SOURCE_DIR}/../../casadi/build_madnlp/test/)
set(MADNLP_DIR ${PROJECT_SOURCE_DIR}/../build/madnlp_c-0.8.3-1.0.0/)
set(JULIA_DIR ${HOME}/.julia/juliaup/julia-1.10.4+0.x64.linux.gnu)

set(JULIA_INCLUDE_DIR ${JULIA_DIR}/include/julia/)
set(CASADI_INCLUDE_DIR ${CASADI_DIR}/include)
set(MADNLP_INCLUDE_DIR ${MADNLP_DIR}/include)

set(CASADI_LIB_DIR ${CASADI_DIR}/lib/)
set(MADNLP_LIB_DIR ${MADNLP_DIR}/lib/)

include(LocalInclude OPTIONAL)

add_compile_options(-Wall -ggdb -std=c++11)

#set(CASADI_IPOPT_SOURCE casadi_ipopt.cpp)
#add_executable(casadi_ipopt ${CASADI_IPOPT_SOURCE})
#target_compile_definitions(casadi_ipopt PUBLIC INCLUDE_DIR=\"${CASADI_INCLUDE_DIR}\")
#target_include_directories(casadi_ipopt PUBLIC {CASADI_INCLUDE_DIR})
#target_link_libraries(casadi_ipopt -L${LIB_DIR} -lcasadi -lstdc++ -ldl -lm)

add_executable(test_madnlp_c main.cpp)
target_compile_definitions(test_madnlp_c PUBLIC INCLUDE_DIR=\"${CASADI_DIR}/include/\")
# target_include_directories(test_madnlp_c PUBLIC ${CASADI_INCLUDE_DIR})
target_include_directories(test_madnlp_c PUBLIC ${MADNLP_INCLUDE_DIR})
target_include_directories(test_madnlp_c PUBLIC ${JULIA_INCLUDE_DIR})
target_link_libraries(test_madnlp_c -L${CASADI_LIB_DIR} -L${MADNLP_LIB_DIR} -lmadnlp_c -ljulia -lstdc++ -ldl -lm)

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/sources/_all/casadi/build_madnlp/test/lib/
