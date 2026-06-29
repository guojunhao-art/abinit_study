#pragma once

#include <Eigen/Dense>
#include <libint2/cxxapi.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace miniqc {

struct OneBodyIntegrals {
    Eigen::MatrixXd S;
    Eigen::MatrixXd T;
    Eigen::MatrixXd V;
    Eigen::MatrixXd Hcore;
};

inline libint2::operator_traits<libint2::Operator::nuclear>::oper_params_type
make_nuclear_operator_params(const std::vector<libint2::Atom>& atoms) {
    libint2::operator_traits<libint2::Operator::nuclear>::oper_params_type q;
    q.reserve(atoms.size());

    for (const libint2::Atom& atom : atoms) {
        q.push_back({
            static_cast<double>(atom.atomic_number),
            {atom.x, atom.y, atom.z}
        });
    }

    return q;
}

inline Eigen::MatrixXd compute_1body_matrix(libint2::Operator op,
                                            const libint2::BasisSet& basis,
                                            const std::vector<libint2::Atom>& atoms) {
    const int nbf = static_cast<int>(basis.nbf());

    libint2::Engine eng(op, basis.max_nprim(), basis.max_l());
    if (op == libint2::Operator::nuclear) {
        eng.set_params(make_nuclear_operator_params(atoms));
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbf, nbf);
    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s1 = 0; s1 < basis.size(); ++s1) {
        for (std::size_t s2 = 0; s2 < basis.size(); ++s2) {
            eng.compute1(basis[s1], basis[s2]);
            const double* buf = eng.results()[0];
            if (!buf) continue;

            const std::size_t bf1 = shell2bf[s1];
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n1 = basis[s1].size();
            const std::size_t n2 = basis[s2].size();

            for (std::size_t i = 0; i < n1; ++i) {
                for (std::size_t j = 0; j < n2; ++j) {
                    M(static_cast<int>(bf1 + i), static_cast<int>(bf2 + j)) =
                        buf[i * n2 + j];
                }
            }
        }
    }

    return 0.5 * (M + M.transpose());
}

inline OneBodyIntegrals build_one_body_integrals(const libint2::BasisSet& basis,
                                                 const std::vector<libint2::Atom>& atoms) {
    OneBodyIntegrals out;
    out.S = compute_1body_matrix(libint2::Operator::overlap, basis, atoms);
    out.T = compute_1body_matrix(libint2::Operator::kinetic, basis, atoms);
    out.V = compute_1body_matrix(libint2::Operator::nuclear, basis, atoms);
    out.Hcore = out.T + out.V;
    return out;
}

}  // namespace miniqc
