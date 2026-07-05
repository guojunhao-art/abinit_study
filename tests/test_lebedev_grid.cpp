#include "lebedev_grid_high_order.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_grid(int n_points, int expected_degree) {
    const auto normalized = dft::lebedev::make_lebedev_grid_extended(
        n_points,
        false  // normalized weights sum to 1
    );
    require(static_cast<int>(normalized.size()) == n_points,
            "Lebedev grid size mismatch");
    require(dft::lebedev::is_supported_extended(n_points),
            "Lebedev grid should be reported as supported");

    const int degree = dft::lebedev::is_supported(n_points)
        ? dft::lebedev::lebedev_degree(n_points)
        : dft::lebedev::high_order_degree(n_points);
    require(degree == expected_degree, "Lebedev degree mismatch");

    double wsum = 0.0;
    for (const auto& p : normalized) {
        require(std::isfinite(p.w), "Lebedev weight should be finite");
        require(std::abs(p.u.norm() - 1.0) < 1.0e-12,
                "Lebedev point should lie on unit sphere");
        wsum += p.w;
    }
    require(std::abs(wsum - 1.0) < 1.0e-12,
            "normalized Lebedev weights should sum to 1");

    const auto four_pi = dft::lebedev::make_lebedev_grid_extended(
        n_points,
        true  // production angular weights sum to 4*pi
    );
    double wsum_four_pi = 0.0;
    for (const auto& p : four_pi) {
        wsum_four_pi += p.w;
    }
    const double target = 4.0 * dft::lebedev::pi();
    require(std::abs(wsum_four_pi - target) < 1.0e-11,
            "4*pi Lebedev weights should sum to 4*pi");
}

}  // namespace

int main() {
    const std::vector<std::pair<int, int>> grids = {
        {6, 3}, {14, 5}, {26, 7}, {38, 9}, {50, 11},
        {74, 13}, {86, 15}, {110, 17}, {146, 19}, {170, 21},
        {194, 23}, {230, 25}, {266, 27}, {302, 29},
        {350, 31}, {434, 35}, {590, 41}, {770, 47},
        {974, 53}, {1202, 59}
    };

    for (const auto& [n, degree] : grids) {
        check_grid(n, degree);
    }

    std::cout << "Lebedev grid test passed\n";
    return 0;
}
