#include "xc_functional.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_pbe() {
    const auto f = miniqc::xc::make_xc_functional("pbe");
    require(f.name == "PBE", "PBE name mismatch");
    require(f.family == miniqc::xc::XCFamily::GGA, "PBE should be GGA");
    require(f.requirements.needs_sigma, "PBE should need sigma");
    require(!f.is_hybrid(), "PBE should not be hybrid");
    require(f.components.size() == 2, "PBE should have two Libxc components");
}

void check_b3lyp() {
    const auto f = miniqc::xc::make_xc_functional("b3lyp");
    require(f.name == "B3LYP", "B3LYP name mismatch");
    require(f.family == miniqc::xc::XCFamily::HybridGGA, "B3LYP should be hybrid GGA");
    require(f.requirements.needs_sigma, "B3LYP should need sigma");
    require(f.is_hybrid(), "B3LYP should be hybrid");
    require(std::abs(f.exact_exchange_fraction - 0.20) < 1.0e-12, "B3LYP exact exchange mismatch");
}

void check_m062x() {
    const auto f = miniqc::xc::make_xc_functional("m06-2x");
    require(f.name == "M06-2X", "M06-2X name mismatch");
    require(f.family == miniqc::xc::XCFamily::HybridMGGA, "M06-2X should be hybrid meta-GGA");
    require(f.requirements.needs_sigma, "M06-2X should need sigma");
    require(f.requirements.needs_tau, "M06-2X should need tau");
    require(f.is_hybrid(), "M06-2X should be hybrid");
    require(std::abs(f.exact_exchange_fraction - 0.54) < 1.0e-12, "M06-2X exact exchange mismatch");
}

}  // namespace

int main() {
    check_pbe();
    check_b3lyp();
    check_m062x();
    std::cout << "XCFunctional smoke test passed\n";
    return 0;
}
