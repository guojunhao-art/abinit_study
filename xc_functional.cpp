#include "xc_functional.hpp"

#include <xc.h>
#include <xc_funcs.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace miniqc::xc {

namespace {

std::string normalize_functional_name(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isspace(c) || c == '_' || c == '-') continue;
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

XCComponent component(int libxc_id, double scale, std::string label) {
    XCComponent c;
    c.libxc_id = libxc_id;
    c.scale = scale;
    c.label = std::move(label);
    return c;
}

void set_gga_requirements(XCFunctional& f) {
    f.requirements.needs_rho = true;
    f.requirements.needs_sigma = true;
}

void set_meta_gga_requirements(XCFunctional& f) {
    f.requirements.needs_rho = true;
    f.requirements.needs_sigma = true;
    f.requirements.needs_tau = true;
}

}  // namespace

bool XCFunctional::is_hybrid() const {
    return exact_exchange_fraction != 0.0 || is_range_separated;
}

bool XCFunctional::is_gga_like() const {
    return requirements.needs_sigma;
}

bool XCFunctional::is_meta_gga() const {
    return requirements.needs_tau || requirements.needs_laplacian;
}

std::string to_string(XCFamily family) {
    switch (family) {
        case XCFamily::LDA: return "LDA";
        case XCFamily::GGA: return "GGA";
        case XCFamily::MGGA: return "MGGA";
        case XCFamily::HybridGGA: return "HybridGGA";
        case XCFamily::HybridMGGA: return "HybridMGGA";
        case XCFamily::RangeSeparatedHybridGGA: return "RangeSeparatedHybridGGA";
        case XCFamily::RangeSeparatedHybridMGGA: return "RangeSeparatedHybridMGGA";
    }
    return "Unknown";
}

std::string to_string(SpinMode spin_mode) {
    switch (spin_mode) {
        case SpinMode::Restricted: return "restricted";
        case SpinMode::Unrestricted: return "unrestricted";
    }
    return "unknown";
}

XCFunctional make_xc_functional(const std::string& name, SpinMode spin_mode) {
    const std::string key = normalize_functional_name(name);
    XCFunctional f;
    f.spin_mode = spin_mode;

    if (key == "slater" || key == "slaterx" || key == "xalpha") {
        f.name = "SlaterX";
        f.family = XCFamily::LDA;
        f.components.push_back(component(XC_LDA_X, 1.0, "XC_LDA_X"));
        return f;
    }
    if (key == "lda" || key == "ldax") {
        f.name = "LDA_X";
        f.family = XCFamily::LDA;
        f.components.push_back(component(XC_LDA_X, 1.0, "XC_LDA_X"));
        return f;
    }
    if (key == "ldapz81" || key == "ldaxpz81" || key == "ldaxpz81c") {
        f.name = "LDA_X_PZ81";
        f.family = XCFamily::LDA;
        f.components.push_back(component(XC_LDA_X, 1.0, "XC_LDA_X"));
        f.components.push_back(component(XC_LDA_C_PZ, 1.0, "XC_LDA_C_PZ"));
        return f;
    }
    if (key == "pbe") {
        f.name = "PBE";
        f.family = XCFamily::GGA;
        set_gga_requirements(f);
        f.components.push_back(component(XC_GGA_X_PBE, 1.0, "XC_GGA_X_PBE"));
        f.components.push_back(component(XC_GGA_C_PBE, 1.0, "XC_GGA_C_PBE"));
        return f;
    }
    if (key == "blyp") {
        f.name = "BLYP";
        f.family = XCFamily::GGA;
        set_gga_requirements(f);
        f.components.push_back(component(XC_GGA_X_B88, 1.0, "XC_GGA_X_B88"));
        f.components.push_back(component(XC_GGA_C_LYP, 1.0, "XC_GGA_C_LYP"));
        return f;
    }
    if (key == "b3lyp") {
        f.name = "B3LYP";
        f.family = XCFamily::HybridGGA;
        set_gga_requirements(f);
        f.exact_exchange_fraction = 0.20;
#ifdef XC_HYB_GGA_XC_B3LYP
        f.components.push_back(component(XC_HYB_GGA_XC_B3LYP, 1.0, "XC_HYB_GGA_XC_B3LYP"));
#else
        throw std::runtime_error("Libxc macro XC_HYB_GGA_XC_B3LYP is unavailable");
#endif
        return f;
    }
    if (key == "pbe0" || key == "pbeh") {
        f.name = "PBE0";
        f.family = XCFamily::HybridGGA;
        set_gga_requirements(f);
        f.exact_exchange_fraction = 0.25;
#ifdef XC_HYB_GGA_XC_PBEH
        f.components.push_back(component(XC_HYB_GGA_XC_PBEH, 1.0, "XC_HYB_GGA_XC_PBEH"));
#else
        throw std::runtime_error("Libxc macro XC_HYB_GGA_XC_PBEH is unavailable");
#endif
        return f;
    }
    if (key == "m062x") {
        f.name = "M06-2X";
        f.family = XCFamily::HybridMGGA;
        set_meta_gga_requirements(f);
        f.exact_exchange_fraction = 0.54;
#ifdef XC_HYB_MGGA_XC_M06_2X
        f.components.push_back(component(XC_HYB_MGGA_XC_M06_2X, 1.0, "XC_HYB_MGGA_XC_M06_2X"));
#else
        throw std::runtime_error("Libxc macro XC_HYB_MGGA_XC_M06_2X is unavailable");
#endif
        return f;
    }

    throw std::runtime_error("unknown XC functional: " + name);
}

bool is_supported_by_current_rks_driver(const XCFunctional& functional) {
    if (functional.is_hybrid()) return false;
    if (functional.is_meta_gga()) return false;
    const std::string key = normalize_functional_name(functional.name);
    return key == "slaterx" || key == "ldax" || key == "ldaxpz81" || key == "pbe";
}

std::string current_rks_support_message(const XCFunctional& functional) {
    if (is_supported_by_current_rks_driver(functional)) {
        return "supported by the current RKS driver";
    }
    std::ostringstream oss;
    oss << functional.name << " is described by XCFunctional but not yet implemented in the current RKS matrix builder.";
    if (functional.is_hybrid()) oss << " It requires exact-exchange K[D].";
    if (functional.requirements.needs_tau) oss << " It requires tau(r) and vtau matrix terms.";
    return oss.str();
}

}  // namespace miniqc::xc
