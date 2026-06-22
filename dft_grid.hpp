#ifndef ABINIT_STUDY_DFT_GRID_HPP
#define ABINIT_STUDY_DFT_GRID_HPP

#include "dft_lda.hpp"

#include <Eigen/Dense>
#include <libint2.hpp>

#include <cstddef>
#include <vector>

namespace dft {

struct AngularGridPoint {
    Eigen::Vector3d u;  // unit direction
    double w = 0.0;     // angular weight, sum_k w_k = 4*pi
};

struct RadialGridPoint {
    double r = 0.0;      // bohr
    double w_rad = 0.0;  // bohr, weight for dr only
};

enum class RadialGridType {
    Uniform,
    PowerMapped
};

enum class PartitionType {
    Distance,
    Becke
};

struct AtomCenteredGridOptions {
    // Number of radial shells around each atom.
    std::size_t n_radial = 80;

    // Supported angular grids in this educational version:
    //   6  : octahedral grid
    //   14 : 6 axis directions + 8 cube-diagonal directions
    //   26 : 6 axis + 12 edge + 8 cube-diagonal directions
    int angular_grid = 14;

    // Radial cutoff in bohr. Increase this for diffuse basis sets.
    double r_max = 12.0;

    // Radial quadrature type.
    RadialGridType radial_type = RadialGridType::PowerMapped;

    // For PowerMapped:
    //   t_i = (i + 1/2) / n_radial
    //   r_i = r_max * t_i^radial_power
    //
    // radial_power > 1 clusters radial shells near the nucleus.
    double radial_power = 2.0;

    // Drop points whose final weight is very small.
    // Usually keep this as zero for debugging electron count.
    double weight_cutoff = 0.0;

    PartitionType partition_type = PartitionType::Becke;

    // Distance partition options:
    //     q_A = 1 / (|r - R_A| + eps)^p
    double partition_eps = 1.0e-10;
    double partition_power = 4.0;

    // Becke partition options:
    int becke_smooth_order = 3;

    bool verbose = true;
};

std::vector<AngularGridPoint>
make_angular_grid(int angular_grid);

std::vector<RadialGridPoint>
make_radial_grid(const AtomCenteredGridOptions& options);

double distance_partition_weight(const std::vector<libint2::Atom>& atoms,
                                 std::size_t atom_index,
                                 const Eigen::Vector3d& r,
                                 double eps,
                                 double power);

double becke_partition_weight(const std::vector<libint2::Atom>& atoms,
                              std::size_t atom_index,
                              const Eigen::Vector3d& r,
                              int smooth_order);

double atom_partition_weight(const std::vector<libint2::Atom>& atoms,
                             std::size_t atom_index,
                             const Eigen::Vector3d& r,
                             const AtomCenteredGridOptions& options);
                             
std::vector<GridPoint>
make_atom_centered_grid(const std::vector<libint2::Atom>& atoms,
                        const AtomCenteredGridOptions& options);


class DIISManager {
public:
    explicit DIISManager(std::size_t max_vectors) : max_vectors_(max_vectors) {
        if (max_vectors_ < 2) {
            throw std::runtime_error("DIISManager: max_vectors must be at least 2");
        }
    }

    std::size_t size() const { return focks_.size(); }

    static Eigen::MatrixXd error_matrix(const Eigen::MatrixXd& F,
                                        const Eigen::MatrixXd& D,
                                        const Eigen::MatrixXd& S,
                                        const Eigen::MatrixXd& X) {
        // Pulay commutator error in an orthonormal AO basis:
        //   e = X^T (F D S - S D F) X
        // For a converged generalized eigenproblem FC=SCe, this vanishes.
        const Eigen::MatrixXd err_ao = F * D * S - S * D * F;
        const Eigen::MatrixXd err = X.transpose() * err_ao * X;
        return 0.5 * (err - err.transpose());
    }

    void add(const Eigen::MatrixXd& F, const Eigen::MatrixXd& err) {
        if (!std::isfinite(err.norm())) {
            throw std::runtime_error("DIISManager: non-finite DIIS error norm");
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

        Eigen::VectorXd coeff =
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
} // namespace dft

#endif // ABINIT_STUDY_DFT_GRID_HPP
