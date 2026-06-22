#include "analytic_gradient.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace miniqc {

namespace {

std::size_t mat_index_4c(std::size_t i,
                         std::size_t j,
                         std::size_t k,
                         std::size_t l,
                         std::size_t n2,
                         std::size_t n3,
                         std::size_t n4) {
    return ((i * n2 + j) * n3 + k) * n4 + l;
}

Eigen::Vector3d atom_position(const libint2::Atom& atom) {
    return Eigen::Vector3d(atom.x, atom.y, atom.z);
}

Eigen::Vector3d shell_center(const libint2::Shell& sh) {
    return Eigen::Vector3d(sh.O[0], sh.O[1], sh.O[2]);
}

std::size_t find_shell_atom(const std::vector<libint2::Atom>& atoms,
                            const libint2::Shell& sh,
                            double tol) {
    const Eigen::Vector3d C = shell_center(sh);

    std::size_t best = atoms.size();
    double best_dist = 1.0e100;

    for (std::size_t A = 0; A < atoms.size(); ++A) {
        const double d = (C - atom_position(atoms[A])).norm();
        if (d < best_dist) {
            best_dist = d;
            best = A;
        }
    }

    if (best == atoms.size() || best_dist > tol) {
        std::ostringstream oss;
        oss << "find_shell_atom: cannot match shell center to an atom. "
            << "best_dist = " << best_dist
            << ", tol = " << tol
            << ", shell center = "
            << C.transpose();
        throw std::runtime_error(oss.str());
    }

    return best;
}

std::size_t mat_index_2c(std::size_t i,
                         std::size_t j,
                         std::size_t n2) {
    return i * n2 + j;
}

std::vector<libint2::Atom>
displaced_atoms(const std::vector<libint2::Atom>& atoms,
                std::size_t atom_index,
                int axis,
                double shift) {
    if (atom_index >= atoms.size()) {
        throw std::runtime_error("displaced_atoms: atom_index out of range");
    }
    if (axis < 0 || axis > 2) {
        throw std::runtime_error("displaced_atoms: axis must be 0, 1, or 2");
    }

    std::vector<libint2::Atom> out = atoms;

    if (axis == 0) out[atom_index].x += shift;
    if (axis == 1) out[atom_index].y += shift;
    if (axis == 2) out[atom_index].z += shift;

    return out;
}

double trace_product(const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("trace_product: dimension mismatch");
    }

    return (A.array() * B.array()).sum();
}

std::vector<std::pair<double, std::array<double, 3>>>
my_make_point_charges(const std::vector<libint2::Atom>& atoms) {
    std::vector<std::pair<double, std::array<double, 3>>> charges;
    charges.reserve(atoms.size());

    for (const auto& atom : atoms) {
        charges.push_back({
            static_cast<double>(atom.atomic_number),
            {{atom.x, atom.y, atom.z}}
        });
    }

    return charges;
}

std::vector<std::pair<double, std::array<double, 3>>>
make_single_point_charge(const libint2::Atom& atom) {
    return {
        {
            static_cast<double>(atom.atomic_number),
            {{atom.x, atom.y, atom.z}}
        }
    };
}

} // namespace


Eigen::MatrixXd
nuclear_repulsion_gradient(const std::vector<libint2::Atom>& atoms) {
    const std::size_t natom = atoms.size();
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(natom), 3
    );

    for (std::size_t A = 0; A < natom; ++A) {
        const double ZA = static_cast<double>(atoms[A].atomic_number);
        const Eigen::Vector3d RA = atom_position(atoms[A]);

        for (std::size_t B = 0; B < natom; ++B) {
            if (A == B) continue;

            const double ZB = static_cast<double>(atoms[B].atomic_number);
            const Eigen::Vector3d RB = atom_position(atoms[B]);

            const Eigen::Vector3d RAB = RA - RB;
            const double R = RAB.norm();

            if (R < 1.0e-12) {
                throw std::runtime_error(
                    "nuclear_repulsion_gradient: two nuclei are too close"
                );
            }

            // d/dR_A [Z_A Z_B / |R_A - R_B|]
            // = - Z_A Z_B (R_A - R_B) / |R_A - R_B|^3
            grad.row(static_cast<Eigen::Index>(A))
                += (-ZA * ZB / (R * R * R) * RAB).transpose();
        }
    }

    return grad;
}


Eigen::MatrixXd
energy_weighted_density(const Eigen::MatrixXd& C,
                        const Eigen::VectorXd& eps,
                        int nocc) {
    if (nocc <= 0) {
        throw std::runtime_error("energy_weighted_density: nocc must be positive");
    }

    const int nbf = static_cast<int>(C.rows());

    if (C.cols() < nocc) {
        throw std::runtime_error("energy_weighted_density: C has too few columns");
    }

    if (eps.size() < nocc) {
        throw std::runtime_error("energy_weighted_density: eps has too few entries");
    }

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nbf, nbf);

    for (int i = 0; i < nocc; ++i) {
        W.noalias() += 2.0 * eps(i) * C.col(i) * C.col(i).transpose();
    }

    return 0.5 * (W + W.transpose());
}


Eigen::MatrixXd
build_overlap_matrix(const libint2::BasisSet& basis) {
    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nbf, nbf);

    libint2::Engine eng(libint2::Operator::overlap,
                        basis.max_nprim(),
                        basis.max_l());

    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s1 = 0; s1 < nshell; ++s1) {
        const std::size_t bf1 = shell2bf[s1];
        const std::size_t n1 = basis[s1].size();

        for (std::size_t s2 = 0; s2 < nshell; ++s2) {
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n2 = basis[s2].size();

            eng.compute(basis[s1], basis[s2]);
            const auto& buf = eng.results();

            if (buf.empty() || buf[0] == nullptr) continue;

            const double* ints = buf[0];

            for (std::size_t i = 0; i < n1; ++i) {
                const Eigen::Index mu = static_cast<Eigen::Index>(bf1 + i);

                for (std::size_t j = 0; j < n2; ++j) {
                    const Eigen::Index nu = static_cast<Eigen::Index>(bf2 + j);

                    S(mu, nu) = ints[mat_index_2c(i, j, n2)];
                }
            }
        }
    }

    return 0.5 * (S + S.transpose());
}

Eigen::MatrixXd
build_kinetic_matrix(const libint2::BasisSet& basis) {
    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();

    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(nbf, nbf);

    libint2::Engine eng(libint2::Operator::kinetic,
                        basis.max_nprim(),
                        basis.max_l());

    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s1 = 0; s1 < nshell; ++s1) {
        const std::size_t bf1 = shell2bf[s1];
        const std::size_t n1 = basis[s1].size();

        for (std::size_t s2 = 0; s2 < nshell; ++s2) {
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n2 = basis[s2].size();

            eng.compute(basis[s1], basis[s2]);
            const auto& buf = eng.results();

            if (buf.empty() || buf[0] == nullptr) continue;

            const double* ints = buf[0];

            for (std::size_t i = 0; i < n1; ++i) {
                const Eigen::Index mu = static_cast<Eigen::Index>(bf1 + i);

                for (std::size_t j = 0; j < n2; ++j) {
                    const Eigen::Index nu = static_cast<Eigen::Index>(bf2 + j);

                    T(mu, nu) = ints[mat_index_2c(i, j, n2)];
                }
            }
        }
    }

    return 0.5 * (T + T.transpose());
}

Eigen::MatrixXd
build_nuclear_attraction_matrix(const std::vector<libint2::Atom>& atoms,
                                const libint2::BasisSet& basis) {
    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(nbf, nbf);

    libint2::Engine eng(libint2::Operator::nuclear,
                        basis.max_nprim(),
                        basis.max_l());

    eng.set_params(my_make_point_charges(atoms));

    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s1 = 0; s1 < nshell; ++s1) {
        const std::size_t bf1 = shell2bf[s1];
        const std::size_t n1 = basis[s1].size();

        for (std::size_t s2 = 0; s2 < nshell; ++s2) {
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n2 = basis[s2].size();

            eng.compute(basis[s1], basis[s2]);
            const auto& buf = eng.results();

            if (buf.empty() || buf[0] == nullptr) continue;

            const double* ints = buf[0];

            for (std::size_t i = 0; i < n1; ++i) {
                const Eigen::Index mu = static_cast<Eigen::Index>(bf1 + i);

                for (std::size_t j = 0; j < n2; ++j) {
                    const Eigen::Index nu = static_cast<Eigen::Index>(bf2 + j);

                    V(mu, nu) = ints[mat_index_2c(i, j, n2)];
                }
            }
        }
    }

    return 0.5 * (V + V.transpose());
}

AtomDerivMatrices
build_overlap_derivative_matrices(const std::vector<libint2::Atom>& atoms,
                                  const libint2::BasisSet& basis,
                                  double center_match_tol) {
    const std::size_t natom = atoms.size();
    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();

    AtomDerivMatrices dS(natom);

    for (std::size_t A = 0; A < natom; ++A) {
        for (int xyz = 0; xyz < 3; ++xyz) {
            dS[A][xyz] = Eigen::MatrixXd::Zero(nbf, nbf);
        }
    }

    std::vector<std::size_t> shell_atom(nshell, 0);

    for (std::size_t s = 0; s < nshell; ++s) {
        shell_atom[s] = find_shell_atom(atoms, basis[s], center_match_tol);
    }

    // First derivative overlap engine.
    //
    // Expected Libint2 result buffer order for 2-center first derivatives:
    //   buf[0] = d/dA_x
    //   buf[1] = d/dA_y
    //   buf[2] = d/dA_z
    //   buf[3] = d/dB_x
    //   buf[4] = d/dB_y
    //   buf[5] = d/dB_z
    //
    // This is why we immediately validate using finite difference below.
    libint2::Engine eng(libint2::Operator::overlap,
                        basis.max_nprim(),
                        basis.max_l(),
                        1);

    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s1 = 0; s1 < nshell; ++s1) {
        const std::size_t bf1 = shell2bf[s1];
        const std::size_t n1 = basis[s1].size();
        const std::size_t A = shell_atom[s1];

        for (std::size_t s2 = 0; s2 < nshell; ++s2) {
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n2 = basis[s2].size();
            const std::size_t B = shell_atom[s2];

            eng.compute(basis[s1], basis[s2]);
            const auto& buf = eng.results();

            if (buf.size() < 6) {
                throw std::runtime_error(
                    "build_overlap_derivative_matrices: expected at least 6 derivative buffers"
                );
            }

            for (int xyz = 0; xyz < 3; ++xyz) {
                const double* buf_A = buf[xyz];
                const double* buf_B = buf[3 + xyz];

                for (std::size_t i = 0; i < n1; ++i) {
                    const Eigen::Index mu = static_cast<Eigen::Index>(bf1 + i);

                    for (std::size_t j = 0; j < n2; ++j) {
                        const Eigen::Index nu = static_cast<Eigen::Index>(bf2 + j);
                        const std::size_t ij = mat_index_2c(i, j, n2);

                        if (buf_A != nullptr) {
                            dS[A][xyz](mu, nu) += buf_A[ij];
                        }

                        if (buf_B != nullptr) {
                            dS[B][xyz](mu, nu) += buf_B[ij];
                        }
                    }
                }
            }
        }
    }

    // Symmetrize each derivative matrix.
    //
    // In exact arithmetic, dS/dR_A should be symmetric because S is symmetric.
    // The shell-pair loop above fills all shell-pair blocks, so this should only
    // remove tiny numerical asymmetry.
    for (std::size_t A = 0; A < natom; ++A) {
        for (int xyz = 0; xyz < 3; ++xyz) {
            dS[A][xyz] = 0.5 * (dS[A][xyz] + dS[A][xyz].transpose());
        }
    }

    return dS;
}

AtomDerivMatrices
build_kinetic_derivative_matrices(const std::vector<libint2::Atom>& atoms,
                                  const libint2::BasisSet& basis,
                                  double center_match_tol) {
    const std::size_t natom = atoms.size();
    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();

    AtomDerivMatrices dT(natom);

    for (std::size_t A = 0; A < natom; ++A) {
        for (int xyz = 0; xyz < 3; ++xyz) {
            dT[A][xyz] = Eigen::MatrixXd::Zero(nbf, nbf);
        }
    }

    std::vector<std::size_t> shell_atom(nshell, 0);

    for (std::size_t s = 0; s < nshell; ++s) {
        shell_atom[s] = find_shell_atom(atoms, basis[s], center_match_tol);
    }

    libint2::Engine eng(libint2::Operator::kinetic,
                        basis.max_nprim(),
                        basis.max_l(),
                        1);

    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s1 = 0; s1 < nshell; ++s1) {
        const std::size_t bf1 = shell2bf[s1];
        const std::size_t n1 = basis[s1].size();
        const std::size_t A = shell_atom[s1];

        for (std::size_t s2 = 0; s2 < nshell; ++s2) {
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n2 = basis[s2].size();
            const std::size_t B = shell_atom[s2];

            eng.compute(basis[s1], basis[s2]);
            const auto& buf = eng.results();

            if (buf.size() < 6) {
                throw std::runtime_error(
                    "build_kinetic_derivative_matrices: expected at least 6 derivative buffers"
                );
            }

            for (int xyz = 0; xyz < 3; ++xyz) {
                const double* buf_A = buf[xyz];
                const double* buf_B = buf[3 + xyz];

                for (std::size_t i = 0; i < n1; ++i) {
                    const Eigen::Index mu = static_cast<Eigen::Index>(bf1 + i);

                    for (std::size_t j = 0; j < n2; ++j) {
                        const Eigen::Index nu = static_cast<Eigen::Index>(bf2 + j);
                        const std::size_t ij = mat_index_2c(i, j, n2);

                        if (buf_A != nullptr) {
                            dT[A][xyz](mu, nu) += buf_A[ij];
                        }

                        if (buf_B != nullptr) {
                            dT[B][xyz](mu, nu) += buf_B[ij];
                        }
                    }
                }
            }
        }
    }

    for (std::size_t A = 0; A < natom; ++A) {
        for (int xyz = 0; xyz < 3; ++xyz) {
            dT[A][xyz] = 0.5 * (dT[A][xyz] + dT[A][xyz].transpose());
        }
    }

    return dT;
}

AtomDerivMatrices
build_nuclear_attraction_derivative_matrices(
    const std::vector<libint2::Atom>& atoms,
    const libint2::BasisSet& basis,
    double center_match_tol
) {
    const std::size_t natom = atoms.size();
    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();

    AtomDerivMatrices dV(natom);

    for (std::size_t A = 0; A < natom; ++A) {
        for (int xyz = 0; xyz < 3; ++xyz) {
            dV[A][xyz] = Eigen::MatrixXd::Zero(nbf, nbf);
        }
    }

    std::vector<std::size_t> shell_atom(nshell, 0);

    for (std::size_t s = 0; s < nshell; ++s) {
        shell_atom[s] = find_shell_atom(atoms, basis[s], center_match_tol);
    }

    const auto& shell2bf = basis.shell2bf();

    // ------------------------------------------------------------
    // Part 1:
    // AO center derivative of the full nuclear attraction operator.
    // ------------------------------------------------------------
    {
        libint2::Engine eng(libint2::Operator::nuclear,
                            basis.max_nprim(),
                            basis.max_l(),
                            1);

        eng.set_params(my_make_point_charges(atoms));

        for (std::size_t s1 = 0; s1 < nshell; ++s1) {
            const std::size_t bf1 = shell2bf[s1];
            const std::size_t n1 = basis[s1].size();
            const std::size_t A = shell_atom[s1];

            for (std::size_t s2 = 0; s2 < nshell; ++s2) {
                const std::size_t bf2 = shell2bf[s2];
                const std::size_t n2 = basis[s2].size();
                const std::size_t B = shell_atom[s2];

                eng.compute(basis[s1], basis[s2]);
                const auto& buf = eng.results();

                if (buf.size() < 6) {
                    throw std::runtime_error(
                        "build_nuclear_attraction_derivative_matrices: expected at least 6 derivative buffers"
                    );
                }

                for (int xyz = 0; xyz < 3; ++xyz) {
                    const double* buf_A = buf[xyz];
                    const double* buf_B = buf[3 + xyz];

                    for (std::size_t i = 0; i < n1; ++i) {
                        const Eigen::Index mu = static_cast<Eigen::Index>(bf1 + i);

                        for (std::size_t j = 0; j < n2; ++j) {
                            const Eigen::Index nu = static_cast<Eigen::Index>(bf2 + j);
                            const std::size_t ij = mat_index_2c(i, j, n2);

                            if (buf_A != nullptr) {
                                dV[A][xyz](mu, nu) += buf_A[ij];
                            }

                            if (buf_B != nullptr) {
                                dV[B][xyz](mu, nu) += buf_B[ij];
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------
    // Part 2:
    // Explicit derivative of the nuclear attraction operator with
    // respect to the nuclear charge positions.
    //
    // For a one-center nuclear attraction operator V_C,
    // translational invariance gives:
    //
    //     dV_C/dR_C = - dV_C/dA - dV_C/dB
    //
    // where A and B are the two Gaussian shell centers.
    // ------------------------------------------------------------
    for (std::size_t C = 0; C < natom; ++C) {
        libint2::Engine engC(libint2::Operator::nuclear,
                             basis.max_nprim(),
                             basis.max_l(),
                             1);

        engC.set_params(make_single_point_charge(atoms[C]));

        for (std::size_t s1 = 0; s1 < nshell; ++s1) {
            const std::size_t bf1 = shell2bf[s1];
            const std::size_t n1 = basis[s1].size();

            for (std::size_t s2 = 0; s2 < nshell; ++s2) {
                const std::size_t bf2 = shell2bf[s2];
                const std::size_t n2 = basis[s2].size();

                engC.compute(basis[s1], basis[s2]);
                const auto& buf = engC.results();

                if (buf.size() < 6) {
                    throw std::runtime_error(
                        "build_nuclear_attraction_derivative_matrices: expected at least 6 derivative buffers for single charge"
                    );
                }

                for (int xyz = 0; xyz < 3; ++xyz) {
                    const double* buf_A = buf[xyz];
                    const double* buf_B = buf[3 + xyz];

                    for (std::size_t i = 0; i < n1; ++i) {
                        const Eigen::Index mu = static_cast<Eigen::Index>(bf1 + i);

                        for (std::size_t j = 0; j < n2; ++j) {
                            const Eigen::Index nu = static_cast<Eigen::Index>(bf2 + j);
                            const std::size_t ij = mat_index_2c(i, j, n2);

                            double explicit_deriv = 0.0;

                            if (buf_A != nullptr) {
                                explicit_deriv -= buf_A[ij];
                            }

                            if (buf_B != nullptr) {
                                explicit_deriv -= buf_B[ij];
                            }

                            dV[C][xyz](mu, nu) += explicit_deriv;
                        }
                    }
                }
            }
        }
    }

    for (std::size_t A = 0; A < natom; ++A) {
        for (int xyz = 0; xyz < 3; ++xyz) {
            dV[A][xyz] = 0.5 * (dV[A][xyz] + dV[A][xyz].transpose());
        }
    }

    return dV;
}

double
rhf_twoelectron_energy_from_eri(const libint2::BasisSet& basis,
                                const Eigen::MatrixXd& D) {
    const int nbf = static_cast<int>(basis.nbf());

    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error("rhf_twoelectron_energy_from_eri: D dimension mismatch");
    }

    const std::size_t nshell = basis.size();

    libint2::Engine eng(libint2::Operator::coulomb,
                        basis.max_nprim(),
                        basis.max_l());

    const auto& shell2bf = basis.shell2bf();

    double E2 = 0.0;

    for (std::size_t s1 = 0; s1 < nshell; ++s1) {
        const std::size_t bf1 = shell2bf[s1];
        const std::size_t n1 = basis[s1].size();

        for (std::size_t s2 = 0; s2 < nshell; ++s2) {
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n2 = basis[s2].size();

            for (std::size_t s3 = 0; s3 < nshell; ++s3) {
                const std::size_t bf3 = shell2bf[s3];
                const std::size_t n3 = basis[s3].size();

                for (std::size_t s4 = 0; s4 < nshell; ++s4) {
                    const std::size_t bf4 = shell2bf[s4];
                    const std::size_t n4 = basis[s4].size();

                    eng.compute(basis[s1], basis[s2], basis[s3], basis[s4]);
                    const auto& buf = eng.results();

                    if (buf.empty() || buf[0] == nullptr) continue;

                    const double* eri = buf[0];

                    for (std::size_t i = 0; i < n1; ++i) {
                        const Eigen::Index mu =
                            static_cast<Eigen::Index>(bf1 + i);

                        for (std::size_t j = 0; j < n2; ++j) {
                            const Eigen::Index nu =
                                static_cast<Eigen::Index>(bf2 + j);

                            for (std::size_t k = 0; k < n3; ++k) {
                                const Eigen::Index lam =
                                    static_cast<Eigen::Index>(bf3 + k);

                                for (std::size_t l = 0; l < n4; ++l) {
                                    const Eigen::Index sig =
                                        static_cast<Eigen::Index>(bf4 + l);

                                    const std::size_t ijkl =
                                        mat_index_4c(i, j, k, l, n2, n3, n4);

                                    const double v = eri[ijkl];

                                    const double Dij = D(mu, nu);
                                    const double Dkl = D(lam, sig);
                                    const double Dik = D(mu, lam);
                                    const double Djl = D(nu, sig);

                                    E2 += 0.5 * Dij * Dkl * v;
                                    E2 += -0.25 * Dik * Djl * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return E2;
}

AtomGradient
build_rhf_twoelectron_derivative_contribution(
    const std::vector<libint2::Atom>& atoms,
    const libint2::BasisSet& basis,
    const Eigen::MatrixXd& D,
    double center_match_tol
) {
    const std::size_t natom = atoms.size();
    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();

    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error(
            "build_rhf_twoelectron_derivative_contribution: D dimension mismatch"
        );
    }

    AtomGradient grad = AtomGradient::Zero(
        static_cast<Eigen::Index>(natom), 3
    );

    std::vector<std::size_t> shell_atom(nshell, 0);

    for (std::size_t s = 0; s < nshell; ++s) {
        shell_atom[s] = find_shell_atom(atoms, basis[s], center_match_tol);
    }

    libint2::Engine eng(libint2::Operator::coulomb,
                        basis.max_nprim(),
                        basis.max_l(),
                        1);

    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s1 = 0; s1 < nshell; ++s1) {
        const std::size_t bf1 = shell2bf[s1];
        const std::size_t n1 = basis[s1].size();
        const std::size_t A1 = shell_atom[s1];

        for (std::size_t s2 = 0; s2 < nshell; ++s2) {
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n2 = basis[s2].size();
            const std::size_t A2 = shell_atom[s2];

            for (std::size_t s3 = 0; s3 < nshell; ++s3) {
                const std::size_t bf3 = shell2bf[s3];
                const std::size_t n3 = basis[s3].size();
                const std::size_t A3 = shell_atom[s3];

                for (std::size_t s4 = 0; s4 < nshell; ++s4) {
                    const std::size_t bf4 = shell2bf[s4];
                    const std::size_t n4 = basis[s4].size();
                    const std::size_t A4 = shell_atom[s4];

                    eng.compute(basis[s1], basis[s2], basis[s3], basis[s4]);
                    const auto& buf = eng.results();

                    if (buf.size() < 12) {
                        throw std::runtime_error(
                            "build_rhf_twoelectron_derivative_contribution: expected 12 derivative buffers"
                        );
                    }

                    for (int center = 0; center < 4; ++center) {
                        std::size_t atom_index = 0;

                        if (center == 0) atom_index = A1;
                        if (center == 1) atom_index = A2;
                        if (center == 2) atom_index = A3;
                        if (center == 3) atom_index = A4;

                        for (int xyz = 0; xyz < 3; ++xyz) {
                            const double* deriv =
                                buf[3 * center + xyz];

                            if (deriv == nullptr) continue;

                            double contribution = 0.0;

                            for (std::size_t i = 0; i < n1; ++i) {
                                const Eigen::Index mu =
                                    static_cast<Eigen::Index>(bf1 + i);

                                for (std::size_t j = 0; j < n2; ++j) {
                                    const Eigen::Index nu =
                                        static_cast<Eigen::Index>(bf2 + j);

                                    for (std::size_t k = 0; k < n3; ++k) {
                                        const Eigen::Index lam =
                                            static_cast<Eigen::Index>(bf3 + k);

                                        for (std::size_t l = 0; l < n4; ++l) {
                                            const Eigen::Index sig =
                                                static_cast<Eigen::Index>(bf4 + l);

                                            const std::size_t ijkl =
                                                mat_index_4c(i, j, k, l, n2, n3, n4);

                                            const double dv = deriv[ijkl];

                                            const double Dij = D(mu, nu);
                                            const double Dkl = D(lam, sig);
                                            const double Dik = D(mu, lam);
                                            const double Djl = D(nu, sig);

                                            contribution += 0.5 * Dij * Dkl * dv;
                                            contribution += -0.25 * Dik * Djl * dv;
                                        }
                                    }
                                }
                            }

                            grad(static_cast<Eigen::Index>(atom_index), xyz)
                                += contribution;
                        }
                    }
                }
            }
        }
    }

    return grad;
}

Eigen::MatrixXd
rhf_analytic_gradient(const std::vector<libint2::Atom>& atoms,
                      const libint2::BasisSet& basis,
                      const Eigen::MatrixXd& D,
                      const Eigen::MatrixXd& C,
                      const Eigen::VectorXd& eps,
                      int nocc,
                      double center_match_tol) {
    const std::size_t natom = atoms.size();
    const int nbf = static_cast<int>(basis.nbf());

    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error("rhf_analytic_gradient: D dimension mismatch");
    }

    if (C.rows() != nbf) {
        throw std::runtime_error("rhf_analytic_gradient: C dimension mismatch");
    }

    if (nocc <= 0 || nocc > C.cols()) {
        throw std::runtime_error("rhf_analytic_gradient: invalid nocc");
    }

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(natom), 3
    );

    // 1. Nuclear repulsion gradient.
    const Eigen::MatrixXd grad_nuc =
        nuclear_repulsion_gradient(atoms);

    // 2. One-electron derivative matrices.
    const AtomDerivMatrices dS =
        build_overlap_derivative_matrices(atoms, basis, center_match_tol);

    const AtomDerivMatrices dT =
        build_kinetic_derivative_matrices(atoms, basis, center_match_tol);

    const AtomDerivMatrices dV =
        build_nuclear_attraction_derivative_matrices(atoms, basis, center_match_tol);

    // 3. RHF two-electron derivative contribution.
    const Eigen::MatrixXd grad_2e =
        build_rhf_twoelectron_derivative_contribution(
            atoms,
            basis,
            D,
            center_match_tol
        );

    // 4. Energy-weighted density matrix.
    const Eigen::MatrixXd W =
        energy_weighted_density(C, eps, nocc);

    // 5. Assemble gradient.
    grad = grad_nuc;

    for (std::size_t A = 0; A < natom; ++A) {
        for (int xyz = 0; xyz < 3; ++xyz) {
            const Eigen::MatrixXd dH = dT[A][xyz] + dV[A][xyz];

            const double one_e =
                trace_product(D, dH);

            const double pulay =
                -trace_product(W, dS[A][xyz]);

            grad(static_cast<Eigen::Index>(A), xyz)
                += one_e
                 + grad_2e(static_cast<Eigen::Index>(A), xyz)
                 + pulay;
        }
    }

    return grad;
}

double
check_overlap_derivative_finite_difference(const std::string& basis_name,
                                           const std::vector<libint2::Atom>& atoms,
                                           std::size_t atom_index,
                                           int axis,
                                           bool force_cartesian,
                                           double h,
                                           bool verbose) {
    if (atoms.empty()) {
        throw std::runtime_error("check_overlap_derivative_finite_difference: no atoms");
    }
    if (atom_index >= atoms.size()) {
        throw std::runtime_error("check_overlap_derivative_finite_difference: atom_index out of range");
    }
    if (axis < 0 || axis > 2) {
        throw std::runtime_error("check_overlap_derivative_finite_difference: axis must be 0, 1, or 2");
    }
    if (!(h > 0.0)) {
        throw std::runtime_error("check_overlap_derivative_finite_difference: h must be positive");
    }

    libint2::BasisSet basis0(basis_name, atoms, true);
    if (force_cartesian) {
        basis0.set_pure(false);
    }

    const AtomDerivMatrices dS = build_overlap_derivative_matrices(atoms, basis0);
    const Eigen::MatrixXd& dS_analytic = dS[atom_index][axis];

    const auto atoms_p = displaced_atoms(atoms, atom_index, axis, +h);
    const auto atoms_m = displaced_atoms(atoms, atom_index, axis, -h);

    libint2::BasisSet basis_p(basis_name, atoms_p, true);
    libint2::BasisSet basis_m(basis_name, atoms_m, true);

    if (force_cartesian) {
        basis_p.set_pure(false);
        basis_m.set_pure(false);
    }

    if (basis_p.nbf() != basis0.nbf() || basis_m.nbf() != basis0.nbf()) {
        throw std::runtime_error(
            "check_overlap_derivative_finite_difference: displaced basis nbf mismatch"
        );
    }

    const Eigen::MatrixXd S_p = build_overlap_matrix(basis_p);
    const Eigen::MatrixXd S_m = build_overlap_matrix(basis_m);

    const Eigen::MatrixXd dS_fd = (S_p - S_m) / (2.0 * h);

    const double max_err = (dS_analytic - dS_fd).cwiseAbs().maxCoeff();
    const double rms_err = std::sqrt(
        (dS_analytic - dS_fd).array().square().mean()
    );

    if (verbose) {
        std::cout << "\n=== overlap derivative finite-difference check ===\n";
        std::cout << "basis           = " << basis_name << "\n";
        std::cout << "force_cartesian = " << std::boolalpha << force_cartesian << "\n";
        std::cout << "atom_index      = " << atom_index << "\n";
        std::cout << "axis            = " << axis << "\n";
        std::cout << "h               = " << std::scientific << h << "\n";
        std::cout << "max_abs_err     = " << std::scientific << std::setprecision(12)
                  << max_err << "\n";
        std::cout << "rms_err         = " << std::scientific << std::setprecision(12)
                  << rms_err << "\n";
    }

    return max_err;
}

double
check_kinetic_derivative_finite_difference(const std::string& basis_name,
                                           const std::vector<libint2::Atom>& atoms,
                                           std::size_t atom_index,
                                           int axis,
                                           bool force_cartesian,
                                           double h,
                                           bool verbose) {
    if (atoms.empty()) {
        throw std::runtime_error("check_kinetic_derivative_finite_difference: no atoms");
    }
    if (atom_index >= atoms.size()) {
        throw std::runtime_error("check_kinetic_derivative_finite_difference: atom_index out of range");
    }
    if (axis < 0 || axis > 2) {
        throw std::runtime_error("check_kinetic_derivative_finite_difference: axis must be 0, 1, or 2");
    }
    if (!(h > 0.0)) {
        throw std::runtime_error("check_kinetic_derivative_finite_difference: h must be positive");
    }

    libint2::BasisSet basis0(basis_name, atoms, true);
    if (force_cartesian) {
        basis0.set_pure(false);
    }

    const AtomDerivMatrices dT = build_kinetic_derivative_matrices(atoms, basis0);
    const Eigen::MatrixXd& dT_analytic = dT[atom_index][axis];

    const auto atoms_p = displaced_atoms(atoms, atom_index, axis, +h);
    const auto atoms_m = displaced_atoms(atoms, atom_index, axis, -h);

    libint2::BasisSet basis_p(basis_name, atoms_p, true);
    libint2::BasisSet basis_m(basis_name, atoms_m, true);

    if (force_cartesian) {
        basis_p.set_pure(false);
        basis_m.set_pure(false);
    }

    if (basis_p.nbf() != basis0.nbf() || basis_m.nbf() != basis0.nbf()) {
        throw std::runtime_error(
            "check_kinetic_derivative_finite_difference: displaced basis nbf mismatch"
        );
    }

    const Eigen::MatrixXd T_p = build_kinetic_matrix(basis_p);
    const Eigen::MatrixXd T_m = build_kinetic_matrix(basis_m);

    const Eigen::MatrixXd dT_fd = (T_p - T_m) / (2.0 * h);

    const double max_err = (dT_analytic - dT_fd).cwiseAbs().maxCoeff();
    const double rms_err = std::sqrt(
        (dT_analytic - dT_fd).array().square().mean()
    );

    if (verbose) {
        std::cout << "\n=== kinetic derivative finite-difference check ===\n";
        std::cout << "basis           = " << basis_name << "\n";
        std::cout << "force_cartesian = " << std::boolalpha << force_cartesian << "\n";
        std::cout << "atom_index      = " << atom_index << "\n";
        std::cout << "axis            = " << axis << "\n";
        std::cout << "h               = " << std::scientific << h << "\n";
        std::cout << "max_abs_err     = " << std::scientific << std::setprecision(12)
                  << max_err << "\n";
        std::cout << "rms_err         = " << std::scientific << std::setprecision(12)
                  << rms_err << "\n";
    }

    return max_err;
}

double
check_nuclear_attraction_derivative_finite_difference(
    const std::string& basis_name,
    const std::vector<libint2::Atom>& atoms,
    std::size_t atom_index,
    int axis,
    bool force_cartesian,
    double h,
    bool verbose
) {
    if (atoms.empty()) {
        throw std::runtime_error("check_nuclear_attraction_derivative_finite_difference: no atoms");
    }
    if (atom_index >= atoms.size()) {
        throw std::runtime_error("check_nuclear_attraction_derivative_finite_difference: atom_index out of range");
    }
    if (axis < 0 || axis > 2) {
        throw std::runtime_error("check_nuclear_attraction_derivative_finite_difference: axis must be 0, 1, or 2");
    }
    if (!(h > 0.0)) {
        throw std::runtime_error("check_nuclear_attraction_derivative_finite_difference: h must be positive");
    }

    libint2::BasisSet basis0(basis_name, atoms, true);
    if (force_cartesian) {
        basis0.set_pure(false);
    }

    const AtomDerivMatrices dV =
        build_nuclear_attraction_derivative_matrices(atoms, basis0);

    const Eigen::MatrixXd& dV_analytic = dV[atom_index][axis];

    const auto atoms_p = displaced_atoms(atoms, atom_index, axis, +h);
    const auto atoms_m = displaced_atoms(atoms, atom_index, axis, -h);

    libint2::BasisSet basis_p(basis_name, atoms_p, true);
    libint2::BasisSet basis_m(basis_name, atoms_m, true);

    if (force_cartesian) {
        basis_p.set_pure(false);
        basis_m.set_pure(false);
    }

    if (basis_p.nbf() != basis0.nbf() || basis_m.nbf() != basis0.nbf()) {
        throw std::runtime_error(
            "check_nuclear_attraction_derivative_finite_difference: displaced basis nbf mismatch"
        );
    }

    const Eigen::MatrixXd V_p = build_nuclear_attraction_matrix(atoms_p, basis_p);
    const Eigen::MatrixXd V_m = build_nuclear_attraction_matrix(atoms_m, basis_m);

    const Eigen::MatrixXd dV_fd = (V_p - V_m) / (2.0 * h);

    const double max_err = (dV_analytic - dV_fd).cwiseAbs().maxCoeff();
    const double rms_err = std::sqrt(
        (dV_analytic - dV_fd).array().square().mean()
    );

    if (verbose) {
        std::cout << "\n=== nuclear attraction derivative finite-difference check ===\n";
        std::cout << "basis           = " << basis_name << "\n";
        std::cout << "force_cartesian = " << std::boolalpha << force_cartesian << "\n";
        std::cout << "atom_index      = " << atom_index << "\n";
        std::cout << "axis            = " << axis << "\n";
        std::cout << "h               = " << std::scientific << h << "\n";
        std::cout << "max_abs_err     = " << std::scientific << std::setprecision(12)
                  << max_err << "\n";
        std::cout << "rms_err         = " << std::scientific << std::setprecision(12)
                  << rms_err << "\n";
    }

    return max_err;
}

double
check_rhf_twoelectron_derivative_finite_difference(
    const std::string& basis_name,
    const std::vector<libint2::Atom>& atoms,
    const Eigen::MatrixXd& D,
    std::size_t atom_index,
    int axis,
    bool force_cartesian,
    double h,
    bool verbose
) {
    if (atoms.empty()) {
        throw std::runtime_error("check_rhf_twoelectron_derivative_finite_difference: no atoms");
    }
    if (atom_index >= atoms.size()) {
        throw std::runtime_error("check_rhf_twoelectron_derivative_finite_difference: atom_index out of range");
    }
    if (axis < 0 || axis > 2) {
        throw std::runtime_error("check_rhf_twoelectron_derivative_finite_difference: axis must be 0, 1, or 2");
    }
    if (!(h > 0.0)) {
        throw std::runtime_error("check_rhf_twoelectron_derivative_finite_difference: h must be positive");
    }

    libint2::BasisSet basis0(basis_name, atoms, true);
    if (force_cartesian) {
        basis0.set_pure(false);
    }

    if (D.rows() != static_cast<int>(basis0.nbf()) ||
        D.cols() != static_cast<int>(basis0.nbf())) {
        throw std::runtime_error(
            "check_rhf_twoelectron_derivative_finite_difference: D dimension mismatch"
        );
    }

    const AtomGradient g2 =
        build_rhf_twoelectron_derivative_contribution(atoms, basis0, D);

    const double analytic =
        g2(static_cast<Eigen::Index>(atom_index), axis);

    const auto atoms_p = displaced_atoms(atoms, atom_index, axis, +h);
    const auto atoms_m = displaced_atoms(atoms, atom_index, axis, -h);

    libint2::BasisSet basis_p(basis_name, atoms_p, true);
    libint2::BasisSet basis_m(basis_name, atoms_m, true);

    if (force_cartesian) {
        basis_p.set_pure(false);
        basis_m.set_pure(false);
    }

    if (basis_p.nbf() != basis0.nbf() ||
        basis_m.nbf() != basis0.nbf()) {
        throw std::runtime_error(
            "check_rhf_twoelectron_derivative_finite_difference: displaced basis nbf mismatch"
        );
    }

    const double E2_p = rhf_twoelectron_energy_from_eri(basis_p, D);
    const double E2_m = rhf_twoelectron_energy_from_eri(basis_m, D);

    const double fd = (E2_p - E2_m) / (2.0 * h);
    const double abs_err = std::abs(analytic - fd);

    if (verbose) {
        std::cout << "\n=== RHF two-electron derivative finite-difference check ===\n";
        std::cout << "basis           = " << basis_name << "\n";
        std::cout << "force_cartesian = " << std::boolalpha << force_cartesian << "\n";
        std::cout << "atom_index      = " << atom_index << "\n";
        std::cout << "axis            = " << axis << "\n";
        std::cout << "h               = " << std::scientific << h << "\n";
        std::cout << "analytic        = " << std::scientific << std::setprecision(12)
                  << analytic << "\n";
        std::cout << "finite diff     = " << std::scientific << std::setprecision(12)
                  << fd << "\n";
        std::cout << "abs_err         = " << std::scientific << std::setprecision(12)
                  << abs_err << "\n";
    }

    return abs_err;
}

} // namespace miniqc
