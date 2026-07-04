#include "xc_evaluator.hpp"
#include "xc_functional.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_finite(double x, const char* message) {
    require(std::isfinite(x), message);
}

void check_lda_point() {
    const auto lda = miniqc::xc::make_xc_functional("lda_x");

    miniqc::xc::XCInputPoint in;
    in.rho = 0.5;

    const auto out = miniqc::xc::evaluate_xc_point(lda, in);
    require_finite(out.exc, "LDA exc should be finite");
    require_finite(out.vrho, "LDA vrho should be finite");
    require(out.exc < 0.0, "LDA exchange exc should be negative");
    require(out.vrho < 0.0, "LDA exchange vrho should be negative");
}

void check_pbe_block() {
    const auto pbe = miniqc::xc::make_xc_functional("pbe");

    miniqc::xc::XCInputBlock in;
    in.rho = {0.2, 0.5, 1.0};
    in.sigma = {0.0, 0.01, 0.04};

    const auto out = miniqc::xc::evaluate_xc_block(pbe, in);
    require(out.exc.size() == in.rho.size(), "PBE output size mismatch");
    require(out.vrho.size() == in.rho.size(), "PBE vrho size mismatch");
    require(out.vsigma.size() == in.rho.size(), "PBE vsigma size mismatch");

    for (std::size_t i = 0; i < in.rho.size(); ++i) {
        require_finite(out.exc[i], "PBE exc should be finite");
        require_finite(out.vrho[i], "PBE vrho should be finite");
        require_finite(out.vsigma[i], "PBE vsigma should be finite");
    }
}

void check_meta_gga_rejected() {
#ifdef XC_HYB_MGGA_XC_M06_2X
    const auto m062x = miniqc::xc::make_xc_functional("m06-2x");

    miniqc::xc::XCInputPoint in;
    in.rho = 0.5;
    in.sigma = 0.01;
    in.tau = 0.2;

    bool threw = false;
    try {
        (void)miniqc::xc::evaluate_xc_point(m062x, in);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "M06-2X should be rejected until tau/meta-GGA evaluator is implemented");
#endif
}

}  // namespace

int main() {
    check_lda_point();
    check_pbe_block();
    check_meta_gga_rejected();
    std::cout << "XCEvaluator smoke test passed\n";
    return 0;
}
