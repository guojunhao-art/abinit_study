#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace miniqc {

inline std::string normalize_element_symbol(std::string symbol) {
    if (symbol.empty()) throw std::runtime_error("empty element symbol");

    for (char& c : symbol) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    symbol[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol[0])));
    return symbol;
}

inline const std::vector<std::string>& element_symbols() {
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
    return symbols;
}

inline int atomic_number_from_token(const std::string& token) {
    if (token.empty()) throw std::runtime_error("empty atom token");

    if (std::isdigit(static_cast<unsigned char>(token[0]))) {
        const int z = std::stoi(token);
        if (z <= 0 || z > 118) throw std::runtime_error("invalid atomic number: " + token);
        return z;
    }

    const std::string symbol = normalize_element_symbol(token);
    const auto& symbols = element_symbols();
    for (std::size_t z = 1; z < symbols.size(); ++z) {
        if (symbols[z] == symbol) return static_cast<int>(z);
    }

    throw std::runtime_error("unknown element symbol: " + token);
}

inline std::string element_symbol_from_z(int z) {
    const auto& symbols = element_symbols();
    if (z <= 0 || z >= static_cast<int>(symbols.size())) {
        return std::to_string(z);
    }
    return symbols[static_cast<std::size_t>(z)];
}

}  // namespace miniqc
