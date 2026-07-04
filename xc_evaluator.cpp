#include "xc_evaluator.hpp"

#include <xc.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace miniqc::xc {

namespace {

class LibxcHandle {
public:
    LibxcHandle(int functional_id, int polarization) {
        const int status = xc_func_init(&func_, functional_id, polarization);
        if (status != 0) {
            std::ostringstream oss;
            oss << "Libxc xc_func_init failed for functional id "
                << functional_id << ", status = " << status;
            throw std::runtime_error(oss.str());
        }
        initialized_ = true;
    }

    LibxcHandle(const LibxcHandle&) = delete;
    LibxcHandle& operator=(const LibxcHandle&) = delete;

    ~LibxcHandle() {
        if (initialized_) {
            xc_func_end(&func_);
        }
    }

    xc_func_type* get() { return &func_; }
    const xc_func_type* get() const { return &func_; }

private:
    xc_func_type func_{};
    bool initialized_ = false;
};

void validate_restricted_input(const XCFunctional& functional,
                               const XCInputBlock& input) {
    if (functional.spin_mode != SpinMode::Restricted) {
        throw std::runtime_error("evaluate_xc_block: only restricted/unpolarized mode is implemented");
    }
    if (input.rho.empty()) {
        throw std::runtime_error("evaluate_xc_block: input.rho is empty");
    }
    if (functional.requirements.needs_sigma && input.sigma.size() != input.rho.size()) {
        throw std::runtime_error("evaluate_xc_block: sigma size does not match rho size");
    }
    if (functional.requirements.needs_laplacian) {
        throw std::runtime_error("evaluate_xc_block: laplacian-dependent functionals are not implemented yet");
    }
    if (functional.requirements.needs_tau) {
        throw std::runtime_error("evaluate_xc_block: tau/meta-GGA functionals are not implemented yet");
    }
}

void accumulate_lda_component(const XCComponent& component,
                              const XCInputBlock& input,
                              XCOutputBlock& output) {
    const std::size_t n = input.size();
    LibxcHandle handle(component.libxc_id, XC_UNPOLARIZED);

    std::vector<double> exc(n, 0.0);
    std::vector<double> vrho(n, 0.0);

    xc_lda_exc_vxc(handle.get(), static_cast<int>(n), input.rho.data(), exc.data(), vrho.data());

    for (std::size_t i = 0; i < n; ++i) {
        output.exc[i] += component.scale * exc[i];
        output.vrho[i] += component.scale * vrho[i];
    }
}

void accumulate_gga_component(const XCComponent& component,
                              const XCInputBlock& input,
                              XCOutputBlock& output) {
    const std::size_t n = input.size();
    LibxcHandle handle(component.libxc_id, XC_UNPOLARIZED);

    std::vector<double> exc(n, 0.0);
    std::vector<double> vrho(n, 0.0);
    std::vector<double> vsigma(n, 0.0);

    xc_gga_exc_vxc(handle.get(), static_cast<int>(n),
                   input.rho.data(), input.sigma.data(),
                   exc.data(), vrho.data(), vsigma.data());

    for (std::size_t i = 0; i < n; ++i) {
        output.exc[i] += component.scale * exc[i];
        output.vrho[i] += component.scale * vrho[i];
        output.vsigma[i] += component.scale * vsigma[i];
    }
}

}  // namespace

void XCOutputBlock::resize(std::size_t n) {
    exc.assign(n, 0.0);
    vrho.assign(n, 0.0);
    vsigma.assign(n, 0.0);
    vlaplacian.assign(n, 0.0);
    vtau.assign(n, 0.0);
}

XCOutputPoint evaluate_xc_point(const XCFunctional& functional,
                                const XCInputPoint& input) {
    XCInputBlock block;
    block.rho = {input.rho};
    if (functional.requirements.needs_sigma) {
        block.sigma = {input.sigma};
    }
    if (functional.requirements.needs_laplacian) {
        block.laplacian = {input.laplacian};
    }
    if (functional.requirements.needs_tau) {
        block.tau = {input.tau};
    }

    const XCOutputBlock out = evaluate_xc_block(functional, block);

    XCOutputPoint point;
    point.exc = out.exc[0];
    point.vrho = out.vrho[0];
    point.vsigma = out.vsigma[0];
    point.vlaplacian = out.vlaplacian[0];
    point.vtau = out.vtau[0];
    return point;
}

XCOutputBlock evaluate_xc_block(const XCFunctional& functional,
                                const XCInputBlock& input) {
    validate_restricted_input(functional, input);

    const std::size_t n = input.size();
    XCOutputBlock output;
    output.resize(n);

    if (functional.components.empty()) {
        throw std::runtime_error("evaluate_xc_block: XCFunctional has no Libxc components");
    }

    for (const XCComponent& component : functional.components) {
        if (functional.requirements.needs_sigma) {
            accumulate_gga_component(component, input, output);
        } else {
            accumulate_lda_component(component, input, output);
        }
    }

    return output;
}

}  // namespace miniqc::xc
