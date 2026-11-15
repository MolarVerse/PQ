/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef CONFIG__ATOM_MASS_MAP_HPP
#define CONFIG__ATOM_MASS_MAP_HPP

#include <map>
#include <string>

#include "unit_types.hpp"

/**
 * @brief Map of atom names to their masses
 *
 * @details special atom names are:
 *
 * d deuterium
 * t tritium
 *
 * q with mass 999.0
 * x with mass 999.0
 * cav with mass 1000.0
 * sup with mass 1000000.0
 * dum with mass 1.0
 */
const std::map<std::string, Mass> atomMassMap = {
    {"h", 1.00794 * mstd::units::amu},
    {"d", 2.014101778 * mstd::units::amu},
    {"t", 3.0160492675 * mstd::units::amu},
    {"he", 4.002602 * mstd::units::amu},
    {"li", 6.941 * mstd::units::amu},
    {"be", 9.012182 * mstd::units::amu},
    {"b", 10.811 * mstd::units::amu},
    {"c", 12.0107 * mstd::units::amu},
    {"n", 14.0067 * mstd::units::amu},
    {"o", 15.9994 * mstd::units::amu},
    {"f", 18.9984032 * mstd::units::amu},
    {"ne", 20.1797 * mstd::units::amu},
    {"na", 22.989770 * mstd::units::amu},
    {"mg", 24.3050 * mstd::units::amu},
    {"al", 26.981538 * mstd::units::amu},
    {"si", 28.0855 * mstd::units::amu},
    {"p", 30.973761 * mstd::units::amu},
    {"s", 32.065 * mstd::units::amu},
    {"cl", 35.453 * mstd::units::amu},
    {"ar", 39.948 * mstd::units::amu},
    {"k", 39.0983 * mstd::units::amu},
    {"ca", 40.078 * mstd::units::amu},
    {"sc", 44.955910 * mstd::units::amu},
    {"ti", 47.880 * mstd::units::amu},
    {"v", 50.9415 * mstd::units::amu},
    {"cr", 51.9961 * mstd::units::amu},
    {"mn", 54.938049 * mstd::units::amu},
    {"fe", 55.845 * mstd::units::amu},
    {"co", 58.933200 * mstd::units::amu},
    {"ni", 58.6934 * mstd::units::amu},
    {"cu", 63.546 * mstd::units::amu},
    {"zn", 65.399 * mstd::units::amu},
    {"ga", 69.723 * mstd::units::amu},
    {"ge", 72.64 * mstd::units::amu},
    {"as", 74.92160 * mstd::units::amu},
    {"se", 78.96 * mstd::units::amu},
    {"br", 79.904 * mstd::units::amu},
    {"kr", 83.798 * mstd::units::amu},
    {"rb", 85.4678 * mstd::units::amu},
    {"sr", 87.62 * mstd::units::amu},
    {"y", 88.90585 * mstd::units::amu},
    {"zr", 91.224 * mstd::units::amu},
    {"nb", 92.90638 * mstd::units::amu},
    {"mo", 95.94 * mstd::units::amu},
    {"tc", 98.9063 * mstd::units::amu},
    {"ru", 101.07 * mstd::units::amu},
    {"rh", 102.9055 * mstd::units::amu},
    {"pd", 106.42 * mstd::units::amu},
    {"ag", 107.8682 * mstd::units::amu},
    {"cd", 112.411 * mstd::units::amu},
    {"in", 114.818 * mstd::units::amu},
    {"sn", 118.71 * mstd::units::amu},
    {"sb", 121.76 * mstd::units::amu},
    {"te", 127.6 * mstd::units::amu},
    {"i", 126.90447 * mstd::units::amu},
    {"xe", 131.293 * mstd::units::amu},
    {"cs", 132.90546 * mstd::units::amu},
    {"ba", 137.327 * mstd::units::amu},
    {"la", 138.9055 * mstd::units::amu},
    {"ce", 140.116 * mstd::units::amu},
    {"pr", 140.90765 * mstd::units::amu},
    {"nd", 144.24 * mstd::units::amu},
    {"pm", 146.9151 * mstd::units::amu},
    {"sm", 150.36 * mstd::units::amu},
    {"lr", 260.1053 * mstd::units::amu},
    {"eu", 151.964 * mstd::units::amu},
    {"gd", 157.25 * mstd::units::amu},
    {"tb", 158.92534 * mstd::units::amu},
    {"dy", 162.5 * mstd::units::amu},
    {"ho", 164.93032 * mstd::units::amu},
    {"er", 167.259 * mstd::units::amu},
    {"tm", 168.93421 * mstd::units::amu},
    {"yb", 173.04 * mstd::units::amu},
    {"lu", 174.967 * mstd::units::amu},
    {"hf", 178.49 * mstd::units::amu},
    {"ta", 180.9479 * mstd::units::amu},
    {"w", 183.84 * mstd::units::amu},
    {"re", 186.207 * mstd::units::amu},
    {"os", 190.23 * mstd::units::amu},
    {"ir", 192.217 * mstd::units::amu},
    {"pt", 195.078 * mstd::units::amu},
    {"au", 196.96655 * mstd::units::amu},
    {"hg", 200.59 * mstd::units::amu},
    {"tl", 204.3833 * mstd::units::amu},
    {"pb", 207.2 * mstd::units::amu},
    {"bi", 208.98038 * mstd::units::amu},
    {"po", 208.9824 * mstd::units::amu},
    {"at", 209.9871 * mstd::units::amu},
    {"rn", 222.0176 * mstd::units::amu},
    {"fr", 223.0197 * mstd::units::amu},
    {"ra", 226.0254 * mstd::units::amu},
    {"ac", 227.0278 * mstd::units::amu},
    {"th", 232.0381 * mstd::units::amu},
    {"pa", 231.03588 * mstd::units::amu},
    {"u", 238.0289 * mstd::units::amu},
    {"np", 237.0482 * mstd::units::amu},
    {"pu", 244.0642 * mstd::units::amu},
    {"am", 243.0614 * mstd::units::amu},
    {"cm", 247.0703 * mstd::units::amu},
    {"bk", 247.0703 * mstd::units::amu},
    {"cf", 251.0796 * mstd::units::amu},
    {"es", 252.0829 * mstd::units::amu},
    {"fm", 257.0951 * mstd::units::amu},
    {"md", 258.0986 * mstd::units::amu},
    {"no", 259.1009 * mstd::units::amu},
    {"q", 999.00000 * mstd::units::amu},
    {"x", 999.00000 * mstd::units::amu},
    {"cav", 1000.00000 * mstd::units::amu},
    {"sup", 1000000.0 * mstd::units::amu},
    {"dum", 1.0 * mstd::units::amu}
};

#endif   // CONFIG__ATOM_MASS_MAP_HPP
