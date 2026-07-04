#include "two_body_fock.hpp"

#include <Eigen/Dense>
#include <libint2/cxxapi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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

        // A simple symmetric test density.  It does not need to be idempotent;
        // this test only verifies the algebraic relation between the builders.
        for (int i = 0; i < nbf; ++i) {
            D(i, i) = 1.0;
        }
        for (int i = 0; i + 1 < nbf; ++i) {
            D(i, i + 1) = 0.1;
            D(i + 1, i) = 0.1;
        }

        const Eigen::MatrixXd J = miniqc::build_j_direct(basis, D);
        const Eigen::MatrixXd K = miniqc::build_k_direct(basis, D);
        const Eigen::MatrixXd G_from_jk = miniqc::build_rhf_g_from_jk(J, K);
        const Eigen::MatrixXd G_direct = miniqc::build_g_direct(basis, D);

        const double err = (G_from_jk - G_direct).norm();
        if (!std::isfinite(err) || err > 1.0e-10) {
            std::cerr << "J/K consistency error = " << err << "\n";
            libint2::finalize();
            return 1;
        }

        std::cout << "J/K builder consistency test passed\n";
        libint2::finalize();
        return 0;
    } catch (const std::exception& e) {
        libint2::finalize();
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
