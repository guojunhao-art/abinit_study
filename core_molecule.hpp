#pragma once

#include <Eigen/Dense>
#include <libint2/cxxapi.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace miniqc {

struct Molecule {
    // Stage-1 refactor keeps libint2::Atom as the stored atom type to avoid
    // touching all existing RHF/gradient/DFT call sites at once.
    std::vector<libint2::Atom> atoms;
    int charge = 0;
    int multiplicity = 1;
};

inline int electron_count(const Molecule& mol) {
    int nelc = -mol.charge;
    for (const auto& atom : mol.atoms) {
        nelc += atom.atomic_number;
    }
    return nelc;
}

inline double nuclear_repulsion_energy(const Molecule& mol) {
    double enuc = 0.0;
    const auto& atoms = mol.atoms;

    for (std::size_t A = 0; A < atoms.size(); ++A) {
        for (std::size_t B = A + 1; B < atoms.size(); ++B) {
            const double ZA = static_cast<double>(atoms[A].atomic_number);
            const double ZB = static_cast<double>(atoms[B].atomic_number);

            const double dx = atoms[A].x - atoms[B].x;
            const double dy = atoms[A].y - atoms[B].y;
            const double dz = atoms[A].z - atoms[B].z;
            const double R = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (R < 1.0e-12) {
                throw std::runtime_error("Two nuclei are too close or overlap.");
            }

            enuc += ZA * ZB / R;
        }
    }

    return enuc;
}

inline Eigen::VectorXd molecule_to_coordinate_vector(const Molecule& mol) {
    Eigen::VectorXd x(3 * static_cast<int>(mol.atoms.size()));
    for (std::size_t a = 0; a < mol.atoms.size(); ++a) {
        x(3 * static_cast<int>(a) + 0) = mol.atoms[a].x;
        x(3 * static_cast<int>(a) + 1) = mol.atoms[a].y;
        x(3 * static_cast<int>(a) + 2) = mol.atoms[a].z;
    }
    return x;
}

inline Molecule molecule_with_coordinate_vector(Molecule mol, const Eigen::VectorXd& x) {
    if (x.size() != 3 * static_cast<int>(mol.atoms.size())) {
        throw std::runtime_error("coordinate vector size does not match molecule");
    }

    for (std::size_t a = 0; a < mol.atoms.size(); ++a) {
        mol.atoms[a].x = x(3 * static_cast<int>(a) + 0);
        mol.atoms[a].y = x(3 * static_cast<int>(a) + 1);
        mol.atoms[a].z = x(3 * static_cast<int>(a) + 2);
    }

    return mol;
}

}  // namespace miniqc
