#include "core_molecule.hpp"
#include "dft_lda.hpp"
#include "one_body_integrals.hpp"
#include "rks_xc.hpp"
#include "xc_functional.hpp"

#include <Eigen/Dense>
#include <libint2/cxxapi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    libint2::initialize();

    try {
        miniqc::Molecule mol;
        mol.atoms = {
            {1, 0.0, 0.0, 0.0},
            {1, 0.0, 0.0, 1.4}
        };

        libint2::BasisSet basis("sto-3g", mol.atoms, true);
        basis.set_pure(false);

        const auto ints = miniqc::build_one_body_integrals(basis, mol.atoms);
        const double Enuc = miniqc::nuclear_repulsion_energy(mol);

        dft::UniformGridOptions grid_options;
        grid_options.spacing = 0.8;
        grid_options.padding = 2.0;
        grid_options.max_points = 200000;
        const auto grid = dft::make_uniform_grid(mol.atoms, grid_options);

        miniqc::rks::RKSXCOptions options;
        options.max_iter = 4;
        options.use_diis = false;
        options.density_mixing = 0.2;
        options.verbose = false;

        const auto pbe = miniqc::xc::make_xc_functional("pbe");
        const auto result = miniqc::rks::run_rks_xc(
            basis, ints.S, ints.Hcore, 2, Enuc, grid, pbe, options);

        require(result.niter > 0, "RKS driver should run at least one iteration");
        require(std::isfinite(result.E_total), "RKS total energy should be finite");
        require(std::isfinite(result.E_xc), "RKS XC energy should be finite");
        require(std::isfinite(result.Ne_grid), "RKS grid electron count should be finite");
        require(result.D.rows() == static_cast<int>(basis.nbf()), "density row mismatch");
        require(result.F.rows() == static_cast<int>(basis.nbf()), "Fock row mismatch");

        std::cout << "Total-XC RKS driver smoke test passed\n";
        libint2::finalize();
        return 0;
    } catch (const std::exception& e) {
        libint2::finalize();
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
