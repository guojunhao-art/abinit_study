cmake_minimum_required(VERSION 3.20)

project(miniqc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MINIQC_BUILD_TESTING "Build and register miniqc regression tests" ON)
option(MINIQC_ENABLE_PYSCF_TESTS "Enable optional PySCF numerical reference tests" OFF)

find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)

if(MINIQC_ENABLE_PYSCF_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

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
    dft_xc_matrix.cpp
    rks_xc.cpp
    xc_functional.cpp
    xc_evaluator.cpp
    dispersion.cpp
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

add_executable(test_xc_functional
    tests/test_xc_functional.cpp
    xc_functional.cpp
)

target_include_directories(test_xc_functional PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBXC_INCLUDE_DIR}
)

target_link_libraries(test_xc_functional PRIVATE
    ${LIBXC_LIBRARY}
)

add_executable(test_xc_evaluator
    tests/test_xc_evaluator.cpp
    xc_functional.cpp
    xc_evaluator.cpp
)

target_include_directories(test_xc_evaluator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBXC_INCLUDE_DIR}
)

target_link_libraries(test_xc_evaluator PRIVATE
    ${LIBXC_LIBRARY}
)

add_executable(test_dft_xc_matrix
    tests/test_dft_xc_matrix.cpp
    dft_grid.cpp
    dft_lda.cpp
    dft_xc_matrix.cpp
    xc_functional.cpp
    xc_evaluator.cpp
)

target_include_directories(test_dft_xc_matrix PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBINT2_INCLUDE_DIR}
    ${LIBXC_INCLUDE_DIR}
)

target_link_libraries(test_dft_xc_matrix PRIVATE
    Eigen3::Eigen
    ${LIBINT2_LIBRARY}
    ${LIBXC_LIBRARY}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_dft_xc_matrix PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_rks_xc_driver
    tests/test_rks_xc_driver.cpp
    dft_grid.cpp
    dft_lda.cpp
    dft_xc_matrix.cpp
    rks_xc.cpp
    xc_functional.cpp
    xc_evaluator.cpp
)

target_include_directories(test_rks_xc_driver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBINT2_INCLUDE_DIR}
    ${LIBXC_INCLUDE_DIR}
)

target_link_libraries(test_rks_xc_driver PRIVATE
    Eigen3::Eigen
    ${LIBINT2_LIBRARY}
    ${LIBXC_LIBRARY}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_rks_xc_driver PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_jk_builders
    tests/test_jk_builders.cpp
)

target_include_directories(test_jk_builders PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBINT2_INCLUDE_DIR}
)

target_link_libraries(test_jk_builders PRIVATE
    Eigen3::Eigen
    ${LIBINT2_LIBRARY}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_jk_builders PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_lebedev_grid
    tests/test_lebedev_grid.cpp
)

target_include_directories(test_lebedev_grid PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(test_lebedev_grid PRIVATE
    Eigen3::Eigen
)

add_executable(test_dispersion_d2
    tests/test_dispersion_d2.cpp
    dispersion.cpp
)

target_include_directories(test_dispersion_d2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBINT2_INCLUDE_DIR}
)

target_link_libraries(test_dispersion_d2 PRIVATE
    Eigen3::Eigen
    ${LIBINT2_LIBRARY}
)

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
            NAME miniqc_xc_functional_smoke
            COMMAND $<TARGET_FILE:test_xc_functional>
        )
        set_tests_properties(miniqc_xc_functional_smoke PROPERTIES
            PASS_REGULAR_EXPRESSION "XCFunctional smoke test passed"
        )

        add_test(
            NAME miniqc_xc_evaluator_smoke
            COMMAND $<TARGET_FILE:test_xc_evaluator>
        )
        set_tests_properties(miniqc_xc_evaluator_smoke PROPERTIES
            PASS_REGULAR_EXPRESSION "XCEvaluator smoke test passed"
        )

        add_test(
            NAME miniqc_dft_xc_matrix_builder
            COMMAND $<TARGET_FILE:test_dft_xc_matrix>
        )
        set_tests_properties(miniqc_dft_xc_matrix_builder PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "DFT XC matrix builder test passed"
        )

        add_test(
            NAME miniqc_rks_xc_driver
            COMMAND $<TARGET_FILE:test_rks_xc_driver>
        )
        set_tests_properties(miniqc_rks_xc_driver PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "Total-XC RKS driver smoke test passed"
        )

        add_test(
            NAME miniqc_jk_builder_consistency
            COMMAND $<TARGET_FILE:test_jk_builders>
        )
        set_tests_properties(miniqc_jk_builder_consistency PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "J/K builder consistency test passed"
        )

        add_test(
            NAME miniqc_lebedev_grid
            COMMAND $<TARGET_FILE:test_lebedev_grid>
        )
        set_tests_properties(miniqc_lebedev_grid PROPERTIES
            PASS_REGULAR_EXPRESSION "Lebedev grid test passed"
        )

        add_test(
            NAME miniqc_dispersion_d2
            COMMAND $<TARGET_FILE:test_dispersion_d2>
        )
        set_tests_properties(miniqc_dispersion_d2 PROPERTIES
            PASS_REGULAR_EXPRESSION "D2 dispersion test passed"
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
            NAME miniqc_h2_pbe_rks_sto3g
            COMMAND $<TARGET_FILE:RHF> ${CMAKE_CURRENT_SOURCE_DIR}/tests/h2_pbe_rks_sto3g.in
        )
        set_tests_properties(miniqc_h2_pbe_rks_sto3g PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "=== RKS/DFT result ==="
        )

        add_test(
            NAME miniqc_h2_pbe_d2_rks_sto3g
            COMMAND $<TARGET_FILE:RHF> ${CMAKE_CURRENT_SOURCE_DIR}/tests/h2_pbe_d2_rks_sto3g.in
        )
        set_tests_properties(miniqc_h2_pbe_d2_rks_sto3g PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "E_dispersion"
        )

        add_test(
            NAME miniqc_h2_b3lyp_rks_sto3g
            COMMAND $<TARGET_FILE:RHF> ${CMAKE_CURRENT_SOURCE_DIR}/tests/h2_b3lyp_rks_sto3g.in
        )
        set_tests_properties(miniqc_h2_b3lyp_rks_sto3g PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "=== RKS/DFT result ==="
        )

        add_test(
            NAME miniqc_h2_pbe0_rks_sto3g
            COMMAND $<TARGET_FILE:RHF> ${CMAKE_CURRENT_SOURCE_DIR}/tests/h2_pbe0_rks_sto3g.in
        )
        set_tests_properties(miniqc_h2_pbe0_rks_sto3g PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "=== RKS/DFT result ==="
        )

        add_test(
            NAME miniqc_h2_m062x_rks_sto3g
            COMMAND $<TARGET_FILE:RHF> ${CMAKE_CURRENT_SOURCE_DIR}/tests/h2_m062x_rks_sto3g.in
        )
        set_tests_properties(miniqc_h2_m062x_rks_sto3g PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
            ENVIRONMENT "OMP_NUM_THREADS=1"
            PASS_REGULAR_EXPRESSION "=== RKS/DFT result ==="
        )

        if(MINIQC_ENABLE_PYSCF_TESTS)
            add_test(
                NAME miniqc_pyscf_dft_reference
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_dft_with_pyscf.py
                    --miniqc-exe $<TARGET_FILE:RHF>
                    --workdir ${CMAKE_CURRENT_BINARY_DIR}/pyscf_reference_work
                    --functionals pbe b3lyp pbe0
                    --n-radial 200
                    --angular-grid 302
                    --tolerance 1.0e-6
            )
            set_tests_properties(miniqc_pyscf_dft_reference PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1"
                PASS_REGULAR_EXPRESSION "PySCF DFT reference comparison passed"
            )
        endif()

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
