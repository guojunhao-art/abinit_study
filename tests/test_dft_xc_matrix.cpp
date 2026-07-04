#include "dft_lda.hpp"
#include "dft_xc_matrix.hpp"
#include "xc_functional.hpp"

#include <Eigen/Dense>
#include <libint2/cxxapi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require_close(double value,
                   double reference,
                   double tol,
                   const char* label) {
    const double err = std::abs(value - reference);
    if (!std::isfinite(err) || err > tol) {
        std::cerr << label << " mismatch: value = " << value
                  << ", reference = " << reference
                  << ", abs_err = " << err << "\n";
        throw std::runtime_error(label);
    }
}

void require_matrix_close(const Eigen::MatrixXd& value,
                          const Eigen::MatrixXd& reference,
                          double tol,
                          const char* label) {
    const double err = (value - reference).norm();
    if (!std::isfinite(err) || err > tol) {
        std::cerr << label << " matrix mismatch: norm_err = " << err << "\n";
        throw std::runtime_error(label);
    }
}

std::vector<dft::GridPoint> make_tiny_test_grid() {
    return {
        {Eigen::Vector3d{0.0, 0.0, 0.0}, 0.25},
        {Eigen::Vector3d{0.0, 0.0, 1.4}, 0.25},
        {Eigen::Vector3d{0.0, 0.0, 0.7}, 0.25},
        {Eigen::Vector3d{0.2, 0.1, 0.6}, 0.25}
    };
}

}  // namespace

int main() {
    libint2::initialize();

    try {
        std::vector<libint2::Atom> atoms = {
            {1, 0.0, 0.0, 0.0},
            {1, 0.0, 0.0, 1.4}
        };

        libint2::BasisSet basis("sto-3g", atoms, true);
        basis.set_pure(false);

        const int nbf = static_cast<int>(basis.nbf());
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(nbf, nbf);
        for (int i = 0; i < nbf; ++i) D(i, i) = 1.0;
        for (int i = 0; i + 1 < nbf; ++i) {
            D(i, i + 1) = 0.1;
            D(i + 1, i) = 0.1;
        }

        const auto grid = make_tiny_test_grid();

        const auto old_lda = dft::build_lda_libxc_sp(
            basis, D, grid, dft::LDAFunctional::Libxc_LDA_X, 1.0e-14);
        const auto new_lda = dft::build_xc_matrix_with_evaluator_sp(
            basis, D, grid, miniqc::xc::make_xc_functional("lda_x"), 1.0e-14);

        require_close(new_lda.Exc, old_lda.Exc, 1.0e-12, "LDA Exc");
        require_close(new_lda.Ne_grid, old_lda.Ne_grid, 1.0e-12, "LDA Ne_grid");
        require_matrix_close(new_lda.Vxc, old_lda.Vxc, 1.0e-12, "LDA Vxc");

        const auto old_pbe = dft::build_gga_libxc_sp(
            basis, D, grid, dft::LDAFunctional::Libxc_GGA_PBE, 1.0e-14);
        const auto new_pbe = dft::build_xc_matrix_with_evaluator_sp(
            basis, D, grid, miniqc::xc::make_xc_functional("pbe"), 1.0e-14);

        require_close(new_pbe.Exc, old_pbe.Exc, 1.0e-12, "PBE Exc");
        require_close(new_pbe.Ne_grid, old_pbe.Ne_grid, 1.0e-12, "PBE Ne_grid");
        require_matrix_close(new_pbe.Vxc, old_pbe.Vxc, 1.0e-12, "PBE Vxc");

        std::cout << "DFT XC matrix builder test passed\n";
        libint2::finalize();
        return 0;
    } catch (const std::exception& e) {
        libint2::finalize();
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
