#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <cstddef>
#include <libint2/cxxapi.h>
#include <vector>

namespace miniqc {

class Tensor4D {
public:
    Tensor4D() = default;
    Tensor4D(std::size_t n0, std::size_t n1, std::size_t n2, std::size_t n3);

    std::size_t dim0() const { return n0_; }
    std::size_t dim1() const { return n1_; }
    std::size_t dim2() const { return n2_; }
    std::size_t dim3() const { return n3_; }
    std::size_t size() const { return data_.size(); }

    double& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
        return data_[index(i, j, k, l)];
    }

    double operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
        return data_[index(i, j, k, l)];
    }

    std::size_t index(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
        return (((i * n1_ + j) * n2_ + k) * n3_ + l);
    }

private:
    std::size_t n0_ = 0;
    std::size_t n1_ = 0;
    std::size_t n2_ = 0;
    std::size_t n3_ = 0;
    std::vector<double> data_;
};

struct FCIOptions {
    bool verbose = true;

    // This is a teaching implementation. It builds a dense FCI Hamiltonian and
    // a full MO ERI tensor, so keep it restricted to small orbital spaces.
    std::size_t max_determinants = 5000;
    std::size_t max_nmo = 16;
    double max_tensor_mb = 4096.0;

    // Print the largest CI coefficients of the ground-state vector.
    bool print_largest_coefficients = false;
    std::size_t n_coefficients_to_print = 12;
};

struct FCIResult {
    double electronic_energy = 0.0;
    double total_energy = 0.0;
    std::size_t nmo = 0;
    int nelec = 0;
    int nalpha = 0;
    int nbeta = 0;
    std::size_t ndet = 0;
    Eigen::VectorXd ci_vector;
};

// Build full spatial-orbital MO ERIs (pq|rs) from Libint AO ERIs and MO coefficients.
// This is intended only for small FCI benchmarks.
Tensor4D build_full_mo_eri_from_libint(const libint2::BasisSet& basis,
                                       const Eigen::MatrixXd& C,
                                       const FCIOptions& options = FCIOptions{});

// Closed-shell/Ms=0 FCI in the full determinant space with fixed Nalpha=Nbeta=nelec/2.
// Hcore is AO one-electron matrix T+Vnuc. C is the canonical MO coefficient matrix.
FCIResult compute_fci_energy(const libint2::BasisSet& basis,
                             const Eigen::MatrixXd& Hcore,
                             const Eigen::MatrixXd& C,
                             int nelec,
                             double nuclear_repulsion,
                             const FCIOptions& options = FCIOptions{},
                             int func = 0);//FCI

}  // namespace miniqc
