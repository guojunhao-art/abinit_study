#pragma once

#include "core_molecule.hpp"

#include <libint2/cxxapi.h>

#include <string>
#include <utility>
#include <vector>

namespace miniqc {

struct BasisContext {
    std::string basis_name;
    bool force_cartesian = true;
    libint2::BasisSet basis;

    BasisContext(const Molecule& mol,
                 std::string basis_name_in,
                 bool force_cartesian_in = true)
        : basis_name(std::move(basis_name_in)),
          force_cartesian(force_cartesian_in),
          basis(basis_name, mol.atoms, true) {
        basis.set_pure(!force_cartesian);
    }

    std::size_t nshell() const { return basis.size(); }
    std::size_t nbf() const { return basis.nbf(); }
};

inline libint2::BasisSet make_basis(const Molecule& mol,
                                    const std::string& basis_name,
                                    bool force_cartesian = true) {
    libint2::BasisSet basis(basis_name, mol.atoms, true);
    basis.set_pure(!force_cartesian);
    return basis;
}

}  // namespace miniqc
