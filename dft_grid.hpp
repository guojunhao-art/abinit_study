#ifndef ABINIT_STUDY_DFT_GRID_HPP
#define ABINIT_STUDY_DFT_GRID_HPP

#include "dft_lda.hpp"
#include "scf_diis.hpp"

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

// Backward-compatible alias.  DIIS is an SCF utility, not a DFT-grid utility;
// existing code that still uses dft::DIISManager will continue to compile.
using DIISManager = miniqc::scf::DIISManager;

} // namespace dft

#endif // ABINIT_STUDY_DFT_GRID_HPP
