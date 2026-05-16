// mini_rhf_libint_direct.cpp
// Starting point for a small C++/Libint2 RHF code.
// Compared with main_libint_bck.cpp, this version removes the AO-indexed
// H2-only ERI callback and builds the Coulomb/exchange contribution by shell
// blocks. This makes the code ready for p/d shells and OpenMP parallelization.

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <libint2/cxxapi.h>

#include "geometry_optimizer.hpp"
#include "mp2_v3_direct_t1.hpp"
#include "fci.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace miniqc {


std::string normalize_element_symbol(std::string symbol) {
    if (symbol.empty()) throw std::runtime_error("empty element symbol");

    for (char& c : symbol) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    symbol[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol[0])));
    return symbol;
}

int atomic_number_from_token(const std::string& token) {
    if (token.empty()) throw std::runtime_error("empty atom token");

    if (std::isdigit(static_cast<unsigned char>(token[0]))) {
        const int z = std::stoi(token);
        if (z <= 0 || z > 118) throw std::runtime_error("invalid atomic number: " + token);
        return z;
    }

    static const std::vector<std::string> symbols = {
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };

    const std::string symbol = normalize_element_symbol(token);
    for (size_t z = 1; z < symbols.size(); ++z) {
        if (symbols[z] == symbol) return static_cast<int>(z);
    }

    throw std::runtime_error("unknown element symbol: " + token);
}

std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::vector<libint2::Atom> read_xyz_atoms(const std::string& filename,
                                          const std::string& unit) {
    constexpr double angstrom_to_bohr = 1.8897261254578281;

    const std::string unit_lc = lowercase(unit);
    double factor = 1.0;
    if (unit_lc == "angstrom" || unit_lc == "ang" || unit_lc == "a") {
        factor = angstrom_to_bohr;
    } else if (unit_lc == "bohr" || unit_lc == "au" || unit_lc == "a.u.") {
        factor = 1.0;
    } else {
        throw std::runtime_error("unknown coordinate unit: " + unit +
                                 " ; use angstrom or bohr");
    }

    std::ifstream input(filename);
    if (!input) throw std::runtime_error("cannot open xyz file: " + filename);

    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty xyz file: " + filename);

    std::istringstream first_line(line);
    size_t natom = 0;
    if (!(first_line >> natom) || natom == 0) {
        throw std::runtime_error("first line of xyz file must be the atom count");
    }

    // Standard XYZ comment line. It may be empty, but it must be present.
    if (!std::getline(input, line)) {
        throw std::runtime_error("xyz file is missing the comment line");
    }

    std::vector<libint2::Atom> atoms;
    atoms.reserve(natom);

    for (size_t i = 0; i < natom; ++i) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("xyz file ended before all atoms were read");
        }

        std::istringstream iss(line);
        std::string symbol_or_z;
        double x = 0.0, y = 0.0, z = 0.0;
        if (!(iss >> symbol_or_z >> x >> y >> z)) {
            throw std::runtime_error("invalid atom line in xyz file: " + line);
        }

        atoms.push_back(libint2::Atom{
            atomic_number_from_token(symbol_or_z),
            x * factor,
            y * factor,
            z * factor
        });
    }

    return atoms;
}

void print_atoms_bohr(const std::vector<libint2::Atom>& atoms) {
    std::cout << "Atoms in bohr:\n";
    for (size_t i = 0; i < atoms.size(); ++i) {
        const auto& a = atoms[i];
        std::cout << "  " << std::setw(3) << i
                  << "  Z=" << std::setw(3) << a.atomic_number
                  << "  " << std::setw(16) << std::setprecision(10) << a.x
                  << "  " << std::setw(16) << std::setprecision(10) << a.y
                  << "  " << std::setw(16) << std::setprecision(10) << a.z
                  << "\n";
    }
    std::cout << "\n";
}

struct Molecule {
    std::vector<libint2::Atom> atoms;
    int charge = 0;
    int multiplicity = 1;
};

int electron_count(const Molecule& mol) {
    int nelc = -mol.charge;

    for (const auto& atom : mol.atoms) {
        nelc += atom.atomic_number;
    }

    return nelc;
}

double nuclear_repulsion_energy(const Molecule& mol) {
    double enuc = 0.0;

    const auto& atoms = mol.atoms;

    for (size_t A = 0; A < atoms.size(); ++A) {
        for (size_t B = A + 1; B < atoms.size(); ++B) {
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

Eigen::MatrixXd symmetric_orthogonalization(const Eigen::MatrixXd& S,
                                            double eps = 1.0e-10) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if (es.info() != Eigen::Success) throw std::runtime_error("S diagonalization failed");

    const auto& evals = es.eigenvalues();
    const auto& U = es.eigenvectors();
    Eigen::VectorXd inv_sqrt = Eigen::VectorXd::Zero(evals.size());

    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) < -eps) throw std::runtime_error("S has a negative eigenvalue");
        if (evals(i) > eps) inv_sqrt(i) = 1.0 / std::sqrt(evals(i));
    }
    return U * inv_sqrt.asDiagonal() * U.transpose();
}


Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int nocc) {
    const int nbf = static_cast<int>(C.rows());
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(nbf, nbf);

    for (int i = 0; i < nocc; ++i) {
        D.noalias() += 2.0 * C.col(i) * C.col(i).transpose();
    }

    return 0.5 * (D + D.transpose());
}

double electronic_energy(const Eigen::MatrixXd& D,
                         const Eigen::MatrixXd& Hcore,
                         const Eigen::MatrixXd& F) {
    return 0.5 * (D.array() * (Hcore + F).array()).sum();
}

class DIISManager {
public:
    explicit DIISManager(size_t max_vectors = 8) : max_vectors_(max_vectors) {
        if (max_vectors_ < 2) {
            throw std::runtime_error("DIIS requires at least 2 vectors");
        }
    }

    size_t size() const { return focks_.size(); }

    static Eigen::MatrixXd error_matrix(const Eigen::MatrixXd& F,
                                        const Eigen::MatrixXd& D,
                                        const Eigen::MatrixXd& S,
                                        const Eigen::MatrixXd& X) {
        // Pulay commutator error in an orthonormal AO basis:
        // e = X^T (F D S - S D F) X.
        Eigen::MatrixXd err_ao = F * D * S - S * D * F;
        Eigen::MatrixXd err = X.transpose() * err_ao * X;
        return 0.5 * (err - err.transpose());
    }

    void add(const Eigen::MatrixXd& F, const Eigen::MatrixXd& error) {
        const double norm = error.norm();
        if (!std::isfinite(norm)) {
            throw std::runtime_error("non-finite DIIS error norm");
        }

        focks_.push_back(F);
        errors_.push_back(error);

        while (focks_.size() > max_vectors_) {
            focks_.erase(focks_.begin());
            errors_.erase(errors_.begin());
        }
    }

    Eigen::MatrixXd extrapolate() const {
        const size_t m = focks_.size();
        if (m < 2) return focks_.back();

        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(static_cast<int>(m + 1),
                                                  static_cast<int>(m + 1));
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<int>(m + 1));

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                B(static_cast<int>(i), static_cast<int>(j)) =
                    (errors_[i].array() * errors_[j].array()).sum();
            }
            B(static_cast<int>(i), static_cast<int>(m)) = -1.0;
            B(static_cast<int>(m), static_cast<int>(i)) = -1.0;
        }
        rhs(static_cast<int>(m)) = -1.0;

        Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
        lu.setThreshold(1.0e-14);
        if (!lu.isInvertible()) {
            // Near-linear dependence between DIIS error vectors. Fall back to latest Fock.
            return focks_.back();
        }

        Eigen::VectorXd sol = lu.solve(rhs);
        if (!sol.allFinite()) return focks_.back();

        Eigen::MatrixXd F_diis = Eigen::MatrixXd::Zero(focks_[0].rows(), focks_[0].cols());
        for (size_t i = 0; i < m; ++i) {
            F_diis.noalias() += sol(static_cast<int>(i)) * focks_[i];
        }

        return 0.5 * (F_diis + F_diis.transpose());
    }

private:
    size_t max_vectors_ = 8;
    std::vector<Eigen::MatrixXd> focks_;
    std::vector<Eigen::MatrixXd> errors_;
};

Eigen::MatrixXd compute_1body_matrix(
    libint2::Operator op,
    const libint2::BasisSet& basis,
    const std::vector<libint2::Atom>& atoms) {

    const int nbf = static_cast<int>(basis.nbf());

    libint2::Engine eng(op, basis.max_nprim(), basis.max_l());
    if (op == libint2::Operator::nuclear) {
        libint2::operator_traits<libint2::Operator::nuclear>::oper_params_type q;
        for (const libint2::Atom& atom : atoms){
            q.push_back({
                static_cast<double>(atom.atomic_number),
                {atom.x, atom.y, atom.z}
            });
        }
        eng.set_params(q);
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(basis.nbf(), basis.nbf());
    const auto& shell2bf = basis.shell2bf();

    for (size_t s1 = 0; s1 < basis.size(); ++s1) {
        for (size_t s2 = 0; s2 < basis.size(); ++s2) {
            eng.compute1(basis[s1], basis[s2]);
            const double* buf = eng.results()[0];
            if (!buf) continue;

            const size_t bf1 = shell2bf[s1];
            const size_t bf2 = shell2bf[s2];
            const size_t n1 = basis[s1].size();
            const size_t n2 = basis[s2].size();

            for (size_t i = 0; i < n1; ++i) {
                for (size_t j = 0; j < n2; ++j) {
                    M(static_cast<int>(bf1 + i), static_cast<int>(bf2 + j)) =
                        buf[i * n2 + j];
                }
            }
        }
    }
    return M;
}

inline size_t eri_index(size_t i, size_t j, size_t k, size_t l,
                        size_t n2, size_t n3, size_t n4) {
    return ((i * n2 + j) * n3 + k) * n4 + l;
}

// Direct shell-block Fock contribution:
// G_mn = sum_ls D_ls [ (mn|ls) - 1/2 (ml|ns) ]
// Density D is spin-summed closed-shell density: D_mn = 2 sum_i C_mi C_ni.
Eigen::MatrixXd build_g_direct(const libint2::BasisSet& basis, const Eigen::MatrixXd& D) {

    const int nbf = static_cast<int>(basis.nbf());

    const size_t nshell = basis.size();

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(basis.nbf(), basis.nbf());

    const long npairs = static_cast<long>(nshell * nshell);

#pragma omp parallel
    {
        libint2::Engine engJ(libint2::Operator::coulomb, basis.max_nprim(), basis.max_l());
        libint2::Engine engK(libint2::Operator::coulomb, basis.max_nprim(), basis.max_l());
        const auto& shell2bf = basis.shell2bf();

#pragma omp for schedule(dynamic)
        for (long pair12 = 0; pair12 < npairs; ++pair12) {
            const size_t s1 = static_cast<size_t>(pair12) / nshell;
            const size_t s2 = static_cast<size_t>(pair12) % nshell;

            const size_t bf1 = shell2bf[s1];
            const size_t bf2 = shell2bf[s2];
            const size_t n1 = basis[s1].size();
            const size_t n2 = basis[s2].size();

            for (size_t s3 = 0; s3 < nshell; ++s3) {
                for (size_t s4 = 0; s4 < nshell; ++s4) {
                    const size_t bf3 = shell2bf[s3];
                    const size_t bf4 = shell2bf[s4];
                    const size_t n3 = basis[s3].size();
                    const size_t n4 = basis[s4].size();

                    engJ.compute2<libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
                        basis[s1], basis[s2], basis[s3], basis[s4]);
                    const double* bufJ = engJ.results()[0];

                    engK.compute2<libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
                        basis[s1], basis[s3], basis[s2], basis[s4]);
                    const double* bufK = engK.results()[0];

                    if (!bufJ && !bufK) continue;

                    for (size_t i = 0; i < n1; ++i) {
                        const int mu = static_cast<int>(bf1 + i);
                        for (size_t j = 0; j < n2; ++j) {
                            const int nu = static_cast<int>(bf2 + j);
                            double gij = 0.0;

                            for (size_t k = 0; k < n3; ++k) {
                                const int lam = static_cast<int>(bf3 + k);
                                for (size_t l = 0; l < n4; ++l) {
                                    const int sig = static_cast<int>(bf4 + l);
                                    const double d_ls = D(lam, sig);
                                    if (std::abs(d_ls) < 1.0e-16) continue;

                                    const double J = bufJ ?
                                        bufJ[eri_index(i, j, k, l, n2, n3, n4)] : 0.0;
                                    // K quartet order is (s1,s3|s2,s4), so local order is (i,k,j,l)
                                    const double K = bufK ?
                                        bufK[eri_index(i, k, j, l, n3, n2, n4)] : 0.0;

                                    gij += d_ls * (J - 0.5 * K);
                                }
                            }
                            G(mu, nu) += gij;
                        }
                    }
                }
            }
        }
    }

    // The all-ordered shell-pair loop should already produce a symmetric G.
    // This removes tiny roundoff asymmetry and is useful before diagonalization.
    return 0.5 * (G + G.transpose());
}

struct RHFResult {
    double energy_electronic = 0.0;
    Eigen::MatrixXd C;
    Eigen::VectorXd eps;
    Eigen::MatrixXd D;
    int iterations = 0;
};

RHFResult rhf_closed_shell(const libint2::BasisSet& basis,
                           const Eigen::MatrixXd& S,
                           const Eigen::MatrixXd& Hcore,
                           int nelec,
                           int max_iter = 128,
                           double e_conv = 1.0e-10,
                           double d_conv = 1.0e-8,
                           bool use_diis = true,
                           int diis_start = 2,
                           size_t diis_max_vec = 8,
                           bool verbose = true,
                           const Eigen::MatrixXd* C_guess = nullptr) {
    if (nelec % 2 != 0) throw std::runtime_error("closed-shell RHF needs even electron count");
    const int nocc = nelec / 2;
    if (nocc <= 0 || nocc > basis.nbf()) throw std::runtime_error("invalid occupied orbital count");

    const int nbf = static_cast<int>(basis.nbf());
    const Eigen::MatrixXd X = symmetric_orthogonalization(S);

    Eigen::MatrixXd D;
    const bool have_valid_C_guess =
        C_guess != nullptr &&
        C_guess->rows() == nbf &&
        C_guess->cols() >= nocc &&
        C_guess->allFinite();

    if (have_valid_C_guess) {
        // Warm start: use converged MO coefficients from a nearby geometry.
        // This is a guess only; it need not be exactly S-orthonormal for the new geometry.
        D = build_density(*C_guess, nocc);
    } else {
        // Core-Hamiltonian initial guess. This is much better than starting from D = 0,
        // and it also gives DIIS a meaningful first commutator error.
        Eigen::MatrixXd Fp0 = X.transpose() * Hcore * X;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es0(Fp0);
        if (es0.info() != Eigen::Success) throw std::runtime_error("core Hamiltonian diagonalization failed");
        Eigen::MatrixXd C0 = X * es0.eigenvectors();
        D = build_density(C0, nocc);
    }

    double E_old = 0.0;
    DIISManager diis(diis_max_vec);
    RHFResult result;

    for (int iter = 1; iter <= max_iter; ++iter) {
        const Eigen::MatrixXd G = build_g_direct(basis, D);
        const Eigen::MatrixXd F_raw = Hcore + G;

        const Eigen::MatrixXd err = DIISManager::error_matrix(F_raw, D, S, X);
        const double diis_err = err.norm();

        Eigen::MatrixXd F_diag = F_raw;
        if (use_diis) {
            diis.add(F_raw, err);
            if (iter >= diis_start && diis.size() >= 2) {
                F_diag = diis.extrapolate();
            }
        }

        const Eigen::MatrixXd Fp = X.transpose() * F_diag * X;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp);
        if (es.info() != Eigen::Success) throw std::runtime_error("Fock diagonalization failed");

        const Eigen::MatrixXd C = X * es.eigenvectors();
        const Eigen::VectorXd eps = es.eigenvalues();
        const Eigen::MatrixXd D_new = build_density(C, nocc);

        // Use the physical Fock F_raw = H + G[D] for the reported electronic energy.
        // F_diag may be a DIIS-extrapolated Fock and is only used to generate orbitals.
        const double E_elec = electronic_energy(D, Hcore, F_raw);
        const double dE = E_elec - E_old;
        const double rmsD = (D_new - D).norm() / static_cast<double>(nbf);

        if (verbose) {
            std::cout << "iter " << std::setw(3) << iter
                      << "  E_elec = " << std::setw(18) << std::setprecision(12) << E_elec
                      << "  dE = " << std::setw(13) << std::scientific << dE
                      << "  rmsD = " << rmsD
                      << "  diis = " << diis_err << std::fixed << "\n";
        }

        result = {E_elec, C, eps, D_new, iter};

        if (iter > 1 && std::abs(dE) < e_conv && rmsD < d_conv) {
            // Rebuild once with the final density to report a strictly consistent energy.
            const Eigen::MatrixXd G_final = build_g_direct(basis, D_new);
            const Eigen::MatrixXd F_final = Hcore + G_final;
            const double E_final = electronic_energy(D_new, Hcore, F_final);

            const Eigen::MatrixXd Fp_final = X.transpose() * F_final * X;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esf(Fp_final);
            if (esf.info() != Eigen::Success) throw std::runtime_error("final Fock diagonalization failed");
            result.energy_electronic = E_final;
            result.C = X * esf.eigenvectors();
            result.eps = esf.eigenvalues();
            result.D = D_new;
            result.iterations = iter;
            return result;
        }

        D = D_new;
        E_old = E_elec;
    }

    throw std::runtime_error("RHF did not converge");
}


Eigen::VectorXd molecule_to_coordinate_vector(const Molecule& mol) {
    Eigen::VectorXd x(3 * static_cast<int>(mol.atoms.size()));
    for (size_t a = 0; a < mol.atoms.size(); ++a) {
        x(3 * static_cast<int>(a) + 0) = mol.atoms[a].x;
        x(3 * static_cast<int>(a) + 1) = mol.atoms[a].y;
        x(3 * static_cast<int>(a) + 2) = mol.atoms[a].z;
    }
    return x;
}

Molecule molecule_with_coordinate_vector(Molecule mol, const Eigen::VectorXd& x) {
    if (x.size() != 3 * static_cast<int>(mol.atoms.size())) {
        throw std::runtime_error("coordinate vector size does not match molecule");
    }
    for (size_t a = 0; a < mol.atoms.size(); ++a) {
        mol.atoms[a].x = x(3 * static_cast<int>(a) + 0);
        mol.atoms[a].y = x(3 * static_cast<int>(a) + 1);
        mol.atoms[a].z = x(3 * static_cast<int>(a) + 2);
    }
    return mol;
}

std::string element_symbol_from_z(int z) {
    static const std::vector<std::string> symbols = {
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };
    if (z <= 0 || z >= static_cast<int>(symbols.size())) {
        return std::to_string(z);
    }
    return symbols[static_cast<size_t>(z)];
}

void write_xyz_angstrom(const std::string& filename,
                        const Molecule& mol,
                        const std::string& comment) {
    constexpr double bohr_to_angstrom = 1.0 / 1.8897261254578281;
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("cannot write xyz file: " + filename);

    out << mol.atoms.size() << "\n";
    out << comment << "\n";
    out << std::fixed << std::setprecision(12);
    for (const auto& atom : mol.atoms) {
        out << std::setw(3) << element_symbol_from_z(atom.atomic_number)
            << "  " << std::setw(18) << atom.x * bohr_to_angstrom
            << "  " << std::setw(18) << atom.y * bohr_to_angstrom
            << "  " << std::setw(18) << atom.z * bohr_to_angstrom
            << "\n";
    }
}

double rhf_total_energy_quiet(const Molecule& mol,
                              const std::string& basis_name,
                              int max_iter = 128,
                              const Eigen::MatrixXd* C_guess = nullptr,
                              Eigen::MatrixXd* C_out = nullptr) {
    if (mol.multiplicity != 1) {
        throw std::runtime_error("closed-shell RHF requires multiplicity 1");
    }
    const int nelec = electron_count(mol);
    if (nelec <= 0) throw std::runtime_error("non-positive electron count");
    if (nelec % 2 != 0) throw std::runtime_error("closed-shell RHF needs an even electron count");

    libint2::BasisSet basis(basis_name, mol.atoms, true);
    Eigen::MatrixXd S = compute_1body_matrix(libint2::Operator::overlap, basis, mol.atoms);
    Eigen::MatrixXd T = compute_1body_matrix(libint2::Operator::kinetic, basis, mol.atoms);
    Eigen::MatrixXd V = compute_1body_matrix(libint2::Operator::nuclear, basis, mol.atoms);
    Eigen::MatrixXd Hcore = T + V;

    RHFResult rhf = rhf_closed_shell(basis, S, Hcore, nelec,
                                     max_iter,
                                     1.0e-10,
                                     1.0e-8,
                                     true,
                                     2,
                                     8,
                                     false,
                                     C_guess);
    if (C_out != nullptr) {
        *C_out = rhf.C;
    }
    return rhf.energy_electronic + nuclear_repulsion_energy(mol);
}

}  // namespace miniqc

int main(int argc, char** argv) {
    libint2::initialize();

    try {
        using namespace miniqc;

        if (argc < 3) {
            std::cerr << "Usage: " << argv[0]
                      << " BASIS_NAME molecule.xyz [unit=angstrom] [charge=0] [multiplicity=1] [--print-matrices] [--mp2] [--fci|--cisd|--cid] [--coeffs] [--opt] [--opt-out file.xyz]\n"
                      << "Example, H2 coordinates in bohr:\n"
                      << "  " << argv[0] << " '6-311g**' h2_bohr.xyz bohr 0 1\n"
                      << "Example, normal XYZ coordinates in angstrom:\n"
                      << "  " << argv[0] << " cc-pvdz water.xyz angstrom 0 1\n";
            libint2::finalize();
            return 1;
        }

        const std::string basis_name = argv[1];
        const std::string xyz_file = argv[2];
        const std::string coord_unit = argc > 3 ? argv[3] : "angstrom";

        Molecule mol;
        mol.atoms = read_xyz_atoms(xyz_file, coord_unit);
        mol.charge = argc > 4 ? std::stoi(argv[4]) : 0;
        mol.multiplicity = argc > 5 ? std::stoi(argv[5]) : 1;

        bool print_matrices = false;
        bool do_mp2 = false;
        bool do_fci = false;
        bool print_fci_coefficients = false;
        bool do_optimization = false;
        int func = 0;
        std::string opt_output_xyz = "optimized.xyz";
        for (int i = 6; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--print-matrices") {
                print_matrices = true;
            } else if (arg == "--mp2") {
                do_mp2 = true;
            } else if (arg == "--fci") {
                if (do_fci == true) throw std::runtime_error("perform only one of fci/cisd/cid each time");
                do_fci = true;
                func = 0;
            } else if (arg == "--cisd") {
                if (do_fci == true) throw std::runtime_error("perform only one of fci/cisd/cid each time");
                do_fci = true;
                func = 1;
            } else if (arg == "--cid") {
                if (do_fci == true) throw std::runtime_error("perform only one of fci/cisd/cid each time");
                do_fci = true;
                func = 2;
            }
              else if (arg == "--coeffs") {
                if (do_fci == false) throw std::runtime_error("--coeffs are only used for fci/cisd/cid");
                print_fci_coefficients = true;
            } else if (arg == "--opt") {
                do_optimization = true;
            } else if (arg == "--opt-out") {
                if (i + 1 >= argc) throw std::runtime_error("--opt-out requires a filename");
                opt_output_xyz = argv[++i];
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        }

        if (mol.multiplicity != 1) {
            throw std::runtime_error("this code currently implements closed-shell RHF only; multiplicity must be 1");
        }

        const int nelec = electron_count(mol);
        if (nelec <= 0) throw std::runtime_error("non-positive electron count");
        if (nelec % 2 != 0) throw std::runtime_error("closed-shell RHF needs an even electron count");

        std::cout << std::fixed << std::setprecision(12);
        std::cout << "=== RHF: Libint2 direct shell-block Fock + DIIS ===\n";
        std::cout << "basis       = " << basis_name << "\n";
        std::cout << "geometry    = " << xyz_file << "\n";
        std::cout << "input unit  = " << coord_unit << "\n";
        std::cout << "charge      = " << mol.charge << "\n";
        std::cout << "multiplicity= " << mol.multiplicity << "\n";
        std::cout << "nelec       = " << nelec << "\n";
        std::cout << "MP2         = " << (do_mp2 ? "yes" : "no") << "\n";
        std::cout << "FCI         = " << (do_fci ? "yes" : "no") << "\n";
#ifdef _OPENMP
        std::cout << "OpenMP threads = " << omp_get_max_threads() << "\n";
#else
        std::cout << "OpenMP disabled\n";
#endif
        std::cout << "\n";

        print_atoms_bohr(mol.atoms);

        libint2::BasisSet basis(basis_name, mol.atoms, true);

        std::cout << "nshell = " << basis.size()
                  << ", nbf = " << basis.nbf()
                  << ", max_nprim = " << basis.max_nprim()
                  << ", max_l = " << basis.max_l()
                  << std::endl;
        const auto& shell2bf = basis.shell2bf();

        for (size_t s = 0; s < basis.size(); ++s) {
            std::cout << "shell " << s
                      << " first AO = " << shell2bf[s]
                      << " size = " << basis[s].size()
                      << " nprim = " << basis[s].nprim()
                      << std::endl;
        }
        std::cout << "\n";

        if (do_optimization && (do_mp2 || do_fci)) {
            throw std::runtime_error("--mp2/--fci are currently implemented for single-point calculations only; run optimization first, then run MP2/FCI on the optimized XYZ file");
        }

        if (do_optimization) {
            GeometryOptions geom_options;
            geom_options.max_steps = 50;
            geom_options.finite_difference_step = 1.0e-3;
            geom_options.max_step = 0.20;
            geom_options.verbose = true;

            const Molecule mol0 = mol;
            const Eigen::VectorXd x0 = molecule_to_coordinate_vector(mol0);

            EnergyFunction energy = [&](const Eigen::VectorXd& x,
                                           const Eigen::MatrixXd* C_guess,
                                           Eigen::MatrixXd* C_out) -> double {
                const Molecule trial = molecule_with_coordinate_vector(mol0, x);
                return rhf_total_energy_quiet(trial, basis_name, 128, C_guess, C_out);
            };

            GeometryResult opt = optimize_bfgs(x0, energy, geom_options);
            Molecule opt_mol = molecule_with_coordinate_vector(mol0, opt.x);

            std::cout << "\n=== Geometry optimization result ===\n";
            std::cout << "converged   = " << (opt.converged ? "yes" : "no") << "\n";
            std::cout << "steps       = " << opt.iterations << "\n";
            std::cout << "E_total     = " << std::setprecision(12) << opt.energy << " Ha\n";
            std::cout << "max|grad|   = " << opt.gradient.cwiseAbs().maxCoeff() << " Ha/bohr\n";
            std::cout << "rms_grad    = " << opt.gradient.norm() / std::sqrt(static_cast<double>(opt.gradient.size())) << " Ha/bohr\n";
            write_xyz_angstrom(opt_output_xyz, opt_mol,
                                "optimized by miniqc RHF numerical-gradient BFGS; coordinates in Angstrom");
            std::cout << "optimized geometry written to " << opt_output_xyz << "\n";

            libint2::finalize();
            return opt.converged ? 0 : 2;
        }

        Eigen::MatrixXd S = compute_1body_matrix(libint2::Operator::overlap, basis, mol.atoms);
        Eigen::MatrixXd T = compute_1body_matrix(libint2::Operator::kinetic, basis, mol.atoms);
        Eigen::MatrixXd V = compute_1body_matrix(libint2::Operator::nuclear, basis, mol.atoms);
        Eigen::MatrixXd Hcore = T + V;

        if (print_matrices) {
            std::cout << "S:\n" << S << "\n\n";
            std::cout << "T:\n" << T << "\n\n";
            std::cout << "V:\n" << V << "\n\n";
            std::cout << "Hcore:\n" << Hcore << "\n\n";
        }

        RHFResult rhf = rhf_closed_shell(basis, S, Hcore, nelec);
        const double Enuc = nuclear_repulsion_energy(mol);
        const double Etot = rhf.energy_electronic + Enuc;

        std::cout << "\n=== Final RHF result ===\n";
        std::cout << "iterations  = " << rhf.iterations << "\n";
        std::cout << "E_elec      = " << rhf.energy_electronic << " Ha\n";
        std::cout << "E_nuc       = " << Enuc << " Ha\n";
        std::cout << "E_total     = " << Etot << " Ha\n\n";
        std::cout << "orbital eps:\n" << rhf.eps.transpose() << "\n\n";

        if (do_mp2) {
            MP2Options mp2_options;
            mp2_options.verbose = true;
            mp2_options.max_intermediate_mb = 8192.0;
            MP2Result mp2 = compute_rhf_mp2_energy(
                basis, rhf.C, rhf.eps, nelec / 2, Etot, mp2_options);

            std::cout << "\n=== RHF-MP2 result ===\n";
            std::cout << "nbf         = " << mp2.nbf << "\n";
            std::cout << "nocc        = " << mp2.nocc << "\n";
            std::cout << "nvir        = " << mp2.nvir << "\n";
            std::cout << std::setprecision(12);
            std::cout << "E_HF        = " << Etot << " Ha\n";
            std::cout << "E_MP2_corr  = " << mp2.correlation_energy << " Ha\n";
            std::cout << "E_MP2_total = " << mp2.total_energy << " Ha\n\n";
        }

        if (do_fci) {
            FCIOptions fci_options;
            fci_options.verbose = true;
            fci_options.max_determinants = 5000;
            fci_options.max_nmo = 160;
            fci_options.max_tensor_mb = 4096.0;
            fci_options.print_largest_coefficients = print_fci_coefficients;

            FCIResult fci = compute_fci_energy(
                basis, Hcore, rhf.C, nelec, Enuc, fci_options, func);
            if (func == 0){    

                std::cout << "\n=== FCI result ===\n";
                std::cout << "nmo         = " << fci.nmo << "\n";
                std::cout << "nelec       = " << fci.nelec << "\n";
                std::cout << "nalpha      = " << fci.nalpha << "\n";
                std::cout << "nbeta       = " << fci.nbeta << "\n";
                std::cout << "ndet        = " << fci.ndet << "\n";
                std::cout << std::setprecision(12);
                std::cout << "E_FCI_elec  = " << fci.electronic_energy << " Ha\n";
                std::cout << "E_nuc       = " << Enuc << " Ha\n";
                std::cout << "E_FCI_total = " << fci.total_energy << " Ha\n\n";
            } else if (func == 1) {
                std::cout << "\n=== CISD result ===\n";
                std::cout << "nmo         = " << fci.nmo << "\n";
                std::cout << "nelec       = " << fci.nelec << "\n";
                std::cout << "nalpha      = " << fci.nalpha << "\n";
                std::cout << "nbeta       = " << fci.nbeta << "\n";
                std::cout << "ndet        = " << fci.ndet << "\n";
                std::cout << std::setprecision(12);
                std::cout << "E_CISD_elec  = " << fci.electronic_energy << " Ha\n";
                std::cout << "E_nuc       = " << Enuc << " Ha\n";
                std::cout << "E_CISD_total = " << fci.total_energy << " Ha\n\n";
            } else if (func == 2){
                std::cout << "\n=== CID result ===\n";
                std::cout << "nmo         = " << fci.nmo << "\n";
                std::cout << "nelec       = " << fci.nelec << "\n";
                std::cout << "nalpha      = " << fci.nalpha << "\n";
                std::cout << "nbeta       = " << fci.nbeta << "\n";
                std::cout << "ndet        = " << fci.ndet << "\n";
                std::cout << std::setprecision(12);
                std::cout << "E_CID_elec  = " << fci.electronic_energy << " Ha\n";
                std::cout << "E_nuc       = " << Enuc << " Ha\n";
                std::cout << "E_CID_total = " << fci.total_energy << " Ha\n\n";
            }
        }

        if (print_matrices) {
            std::cout << "density D:\n" << rhf.D << "\n\n";
            std::cout << "MO C:\n" << rhf.C << "\n";
        }

        libint2::finalize();
        return 0;
    } catch (const std::exception& e) {
        libint2::finalize();
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
