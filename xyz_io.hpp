#pragma once

#include "core_molecule.hpp"
#include "periodic_table.hpp"

#include <libint2/cxxapi.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace miniqc {

inline std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline double coordinate_unit_to_bohr_factor(const std::string& unit) {
    constexpr double angstrom_to_bohr = 1.8897261254578281;

    const std::string unit_lc = lowercase(unit);
    if (unit_lc == "angstrom" || unit_lc == "ang" || unit_lc == "a") {
        return angstrom_to_bohr;
    }
    if (unit_lc == "bohr" || unit_lc == "au" || unit_lc == "a.u.") {
        return 1.0;
    }

    throw std::runtime_error("unknown coordinate unit: " + unit +
                             " ; use angstrom or bohr");
}

inline std::vector<libint2::Atom> read_xyz_atoms(const std::string& filename,
                                                 const std::string& unit) {
    const double factor = coordinate_unit_to_bohr_factor(unit);

    std::ifstream input(filename);
    if (!input) throw std::runtime_error("cannot open xyz file: " + filename);

    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty xyz file: " + filename);

    std::istringstream first_line(line);
    std::size_t natom = 0;
    if (!(first_line >> natom) || natom == 0) {
        throw std::runtime_error("first line of xyz file must be the atom count");
    }

    // Standard XYZ comment line. It may be empty, but it must be present.
    if (!std::getline(input, line)) {
        throw std::runtime_error("xyz file is missing the comment line");
    }

    std::vector<libint2::Atom> atoms;
    atoms.reserve(natom);

    for (std::size_t i = 0; i < natom; ++i) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("xyz file ended before all atoms were read");
        }

        std::istringstream iss(line);
        std::string symbol_or_z;
        double x = 0.0, y = 0.0, z = 0.0;
        if (!(iss >> symbol_or_z >> x >> y >> z)) {
            throw std::runtime_error("invalid atom line in xyz file: " + line);
        }

        atoms.push_back(libint2::Atom{
            atomic_number_from_token(symbol_or_z),
            x * factor,
            y * factor,
            z * factor
        });
    }

    return atoms;
}

inline Molecule read_xyz_molecule(const std::string& filename,
                                  const std::string& unit,
                                  int charge = 0,
                                  int multiplicity = 1) {
    Molecule mol;
    mol.atoms = read_xyz_atoms(filename, unit);
    mol.charge = charge;
    mol.multiplicity = multiplicity;
    return mol;
}

inline void print_atoms_bohr(const std::vector<libint2::Atom>& atoms) {
    std::cout << "Atoms in bohr:\n";
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const auto& a = atoms[i];
        std::cout << "  " << std::setw(3) << i
                  << "  Z=" << std::setw(3) << a.atomic_number
                  << "  " << std::setw(16) << std::setprecision(10) << a.x
                  << "  " << std::setw(16) << std::setprecision(10) << a.y
                  << "  " << std::setw(16) << std::setprecision(10) << a.z
                  << "\n";
    }
    std::cout << "\n";
}

inline void write_xyz_angstrom(const std::string& filename,
                               const Molecule& mol,
                               const std::string& comment) {
    constexpr double bohr_to_angstrom = 1.0 / 1.8897261254578281;

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("cannot write xyz file: " + filename);

    out << mol.atoms.size() << "\n";
    out << comment << "\n";
    out << std::fixed << std::setprecision(12);

    for (const auto& atom : mol.atoms) {
        out << std::setw(3) << element_symbol_from_z(atom.atomic_number)
            << "  " << std::setw(18) << atom.x * bohr_to_angstrom
            << "  " << std::setw(18) << atom.y * bohr_to_angstrom
            << "  " << std::setw(18) << atom.z * bohr_to_angstrom
            << "\n";
    }
}

}  // namespace miniqc
