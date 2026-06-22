#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <libint2/cxxapi.h>

namespace miniqc {

using DerivMatrix3 = std::array<Eigen::MatrixXd, 3>;
using AtomDerivMatrices = std::vector<DerivMatrix3>;

Eigen::MatrixXd
nuclear_repulsion_gradient(const std::vector<libint2::Atom>& atoms);

Eigen::MatrixXd
energy_weighted_density(const Eigen::MatrixXd& C,
                        const Eigen::VectorXd& eps,
                        int nocc);

Eigen::MatrixXd
build_overlap_matrix(const libint2::BasisSet& basis);

AtomDerivMatrices
build_overlap_derivative_matrices(const std::vector<libint2::Atom>& atoms,
                                  const libint2::BasisSet& basis,
                                  double center_match_tol = 1.0e-8);

double
check_overlap_derivative_finite_difference(const std::string& basis_name,
                                           const std::vector<libint2::Atom>& atoms,
                                           std::size_t atom_index,
                                           int axis,
                                           bool force_cartesian,
                                           double h = 1.0e-4,
                                           bool verbose = true);

Eigen::MatrixXd
build_kinetic_matrix(const libint2::BasisSet& basis);

Eigen::MatrixXd
build_nuclear_attraction_matrix(const std::vector<libint2::Atom>& atoms,
                                const libint2::BasisSet& basis);

AtomDerivMatrices
build_kinetic_derivative_matrices(const std::vector<libint2::Atom>& atoms,
                                  const libint2::BasisSet& basis,
                                  double center_match_tol = 1.0e-8);

AtomDerivMatrices
build_nuclear_attraction_derivative_matrices(
    const std::vector<libint2::Atom>& atoms,
    const libint2::BasisSet& basis,
    double center_match_tol = 1.0e-8
);

double
check_kinetic_derivative_finite_difference(const std::string& basis_name,
                                           const std::vector<libint2::Atom>& atoms,
                                           std::size_t atom_index,
                                           int axis,
                                           bool force_cartesian,
                                           double h = 1.0e-4,
                                           bool verbose = true);

double
check_nuclear_attraction_derivative_finite_difference(
    const std::string& basis_name,
    const std::vector<libint2::Atom>& atoms,
    std::size_t atom_index,
    int axis,
    bool force_cartesian,
    double h = 1.0e-4,
    bool verbose = true
);

using AtomGradient = Eigen::MatrixXd;

AtomGradient
build_rhf_twoelectron_derivative_contribution(
    const std::vector<libint2::Atom>& atoms,
    const libint2::BasisSet& basis,
    const Eigen::MatrixXd& D,
    double center_match_tol = 1.0e-8
);

double
rhf_twoelectron_energy_from_eri(const libint2::BasisSet& basis,
                                const Eigen::MatrixXd& D);

double
check_rhf_twoelectron_derivative_finite_difference(
    const std::string& basis_name,
    const std::vector<libint2::Atom>& atoms,
    const Eigen::MatrixXd& D,
    std::size_t atom_index,
    int axis,
    bool force_cartesian,
    double h = 1.0e-4,
    bool verbose = true
);

Eigen::MatrixXd
rhf_analytic_gradient(const std::vector<libint2::Atom>& atoms,
                      const libint2::BasisSet& basis,
                      const Eigen::MatrixXd& D,
                      const Eigen::MatrixXd& C,
                      const Eigen::VectorXd& eps,
                      int nocc,
                      double center_match_tol = 1.0e-8);

double
check_rhf_analytic_gradient_against_finite_difference(
    const std::string& basis_name,
    const std::vector<libint2::Atom>& atoms,
    std::size_t atom_index,
    int axis,
    bool force_cartesian,
    double h,
    bool verbose = true
);


} // namespace miniqc
