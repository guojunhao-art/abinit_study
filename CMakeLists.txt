cmake_minimum_required(VERSION 3.20)

project(miniqc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MINIQC_BUILD_TESTING "Build and register miniqc regression tests" ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)
find_package(Python3 COMPONENTS Interpreter QUIET)

# Libint2 and Libxc are often installed through conda-forge.  The explicit
# find_path/find_library fallback keeps this CMakeLists.txt usable even when
# package config files are not provided by the environment.
find_package(PkgConfig QUIET)

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
        if(NOT Python3_Interpreter_FOUND)
            message(WARNING "Python3 was not found; regression tests will not be registered")
        else()
            add_test(
                NAME miniqc_template_config
                COMMAND $<TARGET_FILE:RHF> --write-template-config
            )

            add_test(
                NAME miniqc_regression_suite
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_regression.py
                    --exe $<TARGET_FILE:RHF>
                    --tests-dir ${CMAKE_CURRENT_SOURCE_DIR}/tests
                    --workdir ${CMAKE_CURRENT_BINARY_DIR}/regression_runs
            )
        endif()
    endif()
endif()
