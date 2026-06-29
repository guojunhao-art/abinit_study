#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace miniqc::scf {

class DIISManager {
public:
    explicit DIISManager(std::size_t max_vectors) : max_vectors_(max_vectors) {
        if (max_vectors_ < 2) {
            throw std::runtime_error("DIISManager: max_vectors must be at least 2");
        }
    }

    std::size_t size() const { return focks_.size(); }

    void clear() {
        focks_.clear();
        errors_.clear();
    }

    static Eigen::MatrixXd error_matrix(const Eigen::MatrixXd& F,
                                        const Eigen::MatrixXd& D,
                                        const Eigen::MatrixXd& S,
                                        const Eigen::MatrixXd& X) {
        // Pulay commutator error in an orthonormal AO basis:
        //   e = X^T (F D S - S D F) X
        // For a converged generalized eigenproblem F C = S C eps, this vanishes.
        const Eigen::MatrixXd err_ao = F * D * S - S * D * F;
        const Eigen::MatrixXd err = X.transpose() * err_ao * X;
        return 0.5 * (err - err.transpose());
    }

    void add(const Eigen::MatrixXd& F, const Eigen::MatrixXd& err) {
        if (!F.allFinite() || !err.allFinite()) {
            throw std::runtime_error("DIISManager: non-finite Fock or error matrix");
        }

        focks_.push_back(F);
        errors_.push_back(err);

        while (focks_.size() > max_vectors_) {
            focks_.erase(focks_.begin());
            errors_.erase(errors_.begin());
        }
    }

    Eigen::MatrixXd extrapolate() const {
        const std::size_t m = focks_.size();
        if (m < 2) {
            throw std::runtime_error("DIISManager: need at least two vectors");
        }

        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(static_cast<int>(m + 1),
                                                  static_cast<int>(m + 1));
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<int>(m + 1));

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                B(static_cast<int>(i), static_cast<int>(j)) =
                    (errors_[i].array() * errors_[j].array()).sum();
            }
            B(static_cast<int>(i), static_cast<int>(m)) = -1.0;
            B(static_cast<int>(m), static_cast<int>(i)) = -1.0;
        }
        rhs(static_cast<int>(m)) = -1.0;

        const Eigen::VectorXd coeff =
            B.completeOrthogonalDecomposition().solve(rhs);

        if ((B * coeff - rhs).norm() > 1.0e-6) {
            throw std::runtime_error("DIISManager: ill-conditioned DIIS solve");
        }

        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(focks_[0].rows(), focks_[0].cols());
        for (std::size_t i = 0; i < m; ++i) {
            F.noalias() += coeff(static_cast<int>(i)) * focks_[i];
        }

        return 0.5 * (F + F.transpose());
    }

private:
    std::size_t max_vectors_;
    std::vector<Eigen::MatrixXd> focks_;
    std::vector<Eigen::MatrixXd> errors_;
};

}  // namespace miniqc::scf
