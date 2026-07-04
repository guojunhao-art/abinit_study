#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

namespace miniqc {

inline Eigen::MatrixXd symmetric_orthogonalization(const Eigen::MatrixXd& S,
                                                   double eps = 1.0e-10) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("symmetric_orthogonalization: S diagonalization failed");
    }

    const auto& evals = es.eigenvalues();
    const auto& U = es.eigenvectors();
    Eigen::VectorXd inv_sqrt = Eigen::VectorXd::Zero(evals.size());

    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) < -eps) {
            throw std::runtime_error("symmetric_orthogonalization: S has a negative eigenvalue");
        }
        if (evals(i) > eps) {
            inv_sqrt(i) = 1.0 / std::sqrt(evals(i));
        }
    }

    return U * inv_sqrt.asDiagonal() * U.transpose();
}

inline Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int nocc) {
    if (nocc <= 0 || C.cols() < nocc) {
        throw std::runtime_error("build_density: invalid occupied orbital count");
    }

    const int nbf = static_cast<int>(C.rows());
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(nbf, nbf);

    for (int i = 0; i < nocc; ++i) {
        D.noalias() += 2.0 * C.col(i) * C.col(i).transpose();
    }

    return 0.5 * (D + D.transpose());
}

inline double trace_product(const Eigen::MatrixXd& A,
                            const Eigen::MatrixXd& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("trace_product: dimension mismatch");
    }
    return (A.array() * B.array()).sum();
}

inline double rhf_electronic_energy(const Eigen::MatrixXd& D,
                                    const Eigen::MatrixXd& Hcore,
                                    const Eigen::MatrixXd& F) {
    return 0.5 * trace_product(D, Hcore + F);
}

inline void diagonalize_fock(const Eigen::MatrixXd& F,
                             const Eigen::MatrixXd& X,
                             Eigen::VectorXd& eps,
                             Eigen::MatrixXd& C) {
    const Eigen::MatrixXd Fp = X.transpose() * F * X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("diagonalize_fock: eigensolver failed");
    }

    eps = es.eigenvalues();
    C = X * es.eigenvectors();
}

}  // namespace miniqc
