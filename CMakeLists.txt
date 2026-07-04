cmake_minimum_required(VERSION 3.20)

project(miniqc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MINIQC_BUILD_TESTING "Build and register miniqc regression tests" ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)

# Libint2 and Libxc are often installed through conda-forge.  The explicit
# find_path/find_library fallback keeps this CMakeLists.txt usable even when
# package config files are not provided by the environment.
find_path(LIBINT2_INCLUDE_DIR
    NAMES libint2/cxxapi.h libint2.hpp
    HINTS $ENV{CONDA_PREFIX}
    PATH_SUFFIXES include
)
find_library(LIBINT2_LIBRARY
    NAMES int2 libint2
    HINTS $ENV{CONDA_PREFIX}
    PATH_SUFFIXES lib
)

find_path(LIBXC_INCLUDE_DIR
    NAMES xc.h
    HINTS $ENV{CONDA_PREFIX}
    PATH_SUFFIXES include
)
find_library(LIBXC_LIBRARY
    NAMES xc libxc
    HINTS $ENV{CONDA_PREFIX}
    PATH_SUFFIXES lib
)

if(NOT LIBINT2_INCLUDE_DIR OR NOT LIBINT2_LIBRARY)
    message(FATAL_ERROR "Could not find Libint2. Set CONDA_PREFIX or pass -DLIBINT2_INCLUDE_DIR=... -DLIBINT2_LIBRARY=...")
endif()

if(NOT LIBXC_INCLUDE_DIR OR NOT LIBXC_LIBRARY)
    message(FATAL_ERROR "Could not find Libxc. Set CONDA_PREFIX or pass -DLIBXC_INCLUDE_DIR=... -DLIBXC_LIBRARY=...")
endif()

set(MINIQC_SOURCES
    main.cpp
    geometry_optimizer.cpp
    analytic_gradient.cpp
    mp2.cpp
    fci.cpp
    dft_grid.cpp
    dft_lda.cpp
    xc_functional.cpp
)

add_executable(RHF ${MINIQC_SOURCES})

target_include_directories(RHF PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBINT2_INCLUDE_DIR}
    ${LIBXC_INCLUDE_DIR}
)

target_link_libraries(RHF PRIVATE
    Eigen3::Eigen
    ${LIBINT2_LIBRARY}
    ${LIBXC_LIBRARY}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(RHF PRIVATE OpenMP::OpenMP_CXX)
endif()

# Keep the historical executable name.  The CMake target is also named RHF, so
# tests and old shell commands can use the same program name.
set_target_properties(RHF PROPERTIES OUTPUT_NAME RHF)

if(MINIQC_BUILD_TESTING)
    include(CTest)
    if(BUILD_TESTING)
        add_test(
            NAME miniqc_template_config
            COMMAND $<TARGET_FILE:RHF> --write-template-config
        )
        set_tests_properties(miniqc_template_config PROPERTIES
            PASS_REGULAR_EXPRESSION "miniqc configuration file"
        )

        add_test(
            NAME miniqc_h2_rhf_sto3g
            COMMAND $<TARGET_FILE:RHF> ${CMAKE_CURRENT_SOURCE_DIR}/tests/h2_rhf_sto3g.in
        )
        set_tests_properties(miniqc_h2_rhf_sto3g PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "E_total[ ]*=[ ]*-1\\.116"
        )

        add_test(
            NAME miniqc_h2_opt_analytic_sto3g
            COMMAND $<TARGET_FILE:RHF> ${CMAKE_CURRENT_SOURCE_DIR}/tests/h2_opt_analytic_sto3g.in
        )
        set_tests_properties(miniqc_h2_opt_analytic_sto3g PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "Geometry optimization result"
        )
    endif()
endif()
