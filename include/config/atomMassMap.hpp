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
    {"h", 1.00794 * mstd::units::g / mstd::units::mol},
    {"d", 2.014101778 * mstd::units::g / mstd::units::mol},
    {"t", 3.0160492675 * mstd::units::g / mstd::units::mol},
    {"he", 4.002602 * mstd::units::g / mstd::units::mol},
    {"li", 6.941 * mstd::units::g / mstd::units::mol},
    {"be", 9.012182 * mstd::units::g / mstd::units::mol},
    {"b", 10.811 * mstd::units::g / mstd::units::mol},
    {"c", 12.0107 * mstd::units::g / mstd::units::mol},
    {"n", 14.0067 * mstd::units::g / mstd::units::mol},
    {"o", 15.9994 * mstd::units::g / mstd::units::mol},
    {"f", 18.9984032 * mstd::units::g / mstd::units::mol},
    {"ne", 20.1797 * mstd::units::g / mstd::units::mol},
    {"na", 22.989770 * mstd::units::g / mstd::units::mol},
    {"mg", 24.3050 * mstd::units::g / mstd::units::mol},
    {"al", 26.981538 * mstd::units::g / mstd::units::mol},
    {"si", 28.0855 * mstd::units::g / mstd::units::mol},
    {"p", 30.973761 * mstd::units::g / mstd::units::mol},
    {"s", 32.065 * mstd::units::g / mstd::units::mol},
    {"cl", 35.453 * mstd::units::g / mstd::units::mol},
    {"ar", 39.948 * mstd::units::g / mstd::units::mol},
    {"k", 39.0983 * mstd::units::g / mstd::units::mol},
    {"ca", 40.078 * mstd::units::g / mstd::units::mol},
    {"sc", 44.955910 * mstd::units::g / mstd::units::mol},
    {"ti", 47.880 * mstd::units::g / mstd::units::mol},
    {"v", 50.9415 * mstd::units::g / mstd::units::mol},
    {"cr", 51.9961 * mstd::units::g / mstd::units::mol},
    {"mn", 54.938049 * mstd::units::g / mstd::units::mol},
    {"fe", 55.845 * mstd::units::g / mstd::units::mol},
    {"co", 58.933200 * mstd::units::g / mstd::units::mol},
    {"ni", 58.6934 * mstd::units::g / mstd::units::mol},
    {"cu", 63.546 * mstd::units::g / mstd::units::mol},
    {"zn", 65.399 * mstd::units::g / mstd::units::mol},
    {"ga", 69.723 * mstd::units::g / mstd::units::mol},
    {"ge", 72.64 * mstd::units::g / mstd::units::mol},
    {"as", 74.92160 * mstd::units::g / mstd::units::mol},
    {"se", 78.96 * mstd::units::g / mstd::units::mol},
    {"br", 79.904 * mstd::units::g / mstd::units::mol},
    {"kr", 83.798 * mstd::units::g / mstd::units::mol},
    {"rb", 85.4678 * mstd::units::g / mstd::units::mol},
    {"sr", 87.62 * mstd::units::g / mstd::units::mol},
    {"y", 88.90585 * mstd::units::g / mstd::units::mol},
    {"zr", 91.224 * mstd::units::g / mstd::units::mol},
    {"nb", 92.90638 * mstd::units::g / mstd::units::mol},
    {"mo", 95.94 * mstd::units::g / mstd::units::mol},
    {"tc", 98.9063 * mstd::units::g / mstd::units::mol},
    {"ru", 101.07 * mstd::units::g / mstd::units::mol},
    {"rh", 102.9055 * mstd::units::g / mstd::units::mol},
    {"pd", 106.42 * mstd::units::g / mstd::units::mol},
    {"ag", 107.8682 * mstd::units::g / mstd::units::mol},
    {"cd", 112.411 * mstd::units::g / mstd::units::mol},
    {"in", 114.818 * mstd::units::g / mstd::units::mol},
    {"sn", 118.71 * mstd::units::g / mstd::units::mol},
    {"sb", 121.76 * mstd::units::g / mstd::units::mol},
    {"te", 127.6 * mstd::units::g / mstd::units::mol},
    {"i", 126.90447 * mstd::units::g / mstd::units::mol},
    {"xe", 131.293 * mstd::units::g / mstd::units::mol},
    {"cs", 132.90546 * mstd::units::g / mstd::units::mol},
    {"ba", 137.327 * mstd::units::g / mstd::units::mol},
    {"la", 138.9055 * mstd::units::g / mstd::units::mol},
    {"ce", 140.116 * mstd::units::g / mstd::units::mol},
    {"pr", 140.90765 * mstd::units::g / mstd::units::mol},
    {"nd", 144.24 * mstd::units::g / mstd::units::mol},
    {"pm", 146.9151 * mstd::units::g / mstd::units::mol},
    {"sm", 150.36 * mstd::units::g / mstd::units::mol},
    {"lr", 260.1053 * mstd::units::g / mstd::units::mol},
    {"eu", 151.964 * mstd::units::g / mstd::units::mol},
    {"gd", 157.25 * mstd::units::g / mstd::units::mol},
    {"tb", 158.92534 * mstd::units::g / mstd::units::mol},
    {"dy", 162.5 * mstd::units::g / mstd::units::mol},
    {"ho", 164.93032 * mstd::units::g / mstd::units::mol},
    {"er", 167.259 * mstd::units::g / mstd::units::mol},
    {"tm", 168.93421 * mstd::units::g / mstd::units::mol},
    {"yb", 173.04 * mstd::units::g / mstd::units::mol},
    {"lu", 174.967 * mstd::units::g / mstd::units::mol},
    {"hf", 178.49 * mstd::units::g / mstd::units::mol},
    {"ta", 180.9479 * mstd::units::g / mstd::units::mol},
    {"w", 183.84 * mstd::units::g / mstd::units::mol},
    {"re", 186.207 * mstd::units::g / mstd::units::mol},
    {"os", 190.23 * mstd::units::g / mstd::units::mol},
    {"ir", 192.217 * mstd::units::g / mstd::units::mol},
    {"pt", 195.078 * mstd::units::g / mstd::units::mol},
    {"au", 196.96655 * mstd::units::g / mstd::units::mol},
    {"hg", 200.59 * mstd::units::g / mstd::units::mol},
    {"tl", 204.3833 * mstd::units::g / mstd::units::mol},
    {"pb", 207.2 * mstd::units::g / mstd::units::mol},
    {"bi", 208.98038 * mstd::units::g / mstd::units::mol},
    {"po", 208.9824 * mstd::units::g / mstd::units::mol},
    {"at", 209.9871 * mstd::units::g / mstd::units::mol},
    {"rn", 222.0176 * mstd::units::g / mstd::units::mol},
    {"fr", 223.0197 * mstd::units::g / mstd::units::mol},
    {"ra", 226.0254 * mstd::units::g / mstd::units::mol},
    {"ac", 227.0278 * mstd::units::g / mstd::units::mol},
    {"th", 232.0381 * mstd::units::g / mstd::units::mol},
    {"pa", 231.03588 * mstd::units::g / mstd::units::mol},
    {"u", 238.0289 * mstd::units::g / mstd::units::mol},
    {"np", 237.0482 * mstd::units::g / mstd::units::mol},
    {"pu", 244.0642 * mstd::units::g / mstd::units::mol},
    {"am", 243.0614 * mstd::units::g / mstd::units::mol},
    {"cm", 247.0703 * mstd::units::g / mstd::units::mol},
    {"bk", 247.0703 * mstd::units::g / mstd::units::mol},
    {"cf", 251.0796 * mstd::units::g / mstd::units::mol},
    {"es", 252.0829 * mstd::units::g / mstd::units::mol},
    {"fm", 257.0951 * mstd::units::g / mstd::units::mol},
    {"md", 258.0986 * mstd::units::g / mstd::units::mol},
    {"no", 259.1009 * mstd::units::g / mstd::units::mol},
    {"q", 999.00000 * mstd::units::g / mstd::units::mol},
    {"x", 999.00000 * mstd::units::g / mstd::units::mol},
    {"cav", 1000.00000 * mstd::units::g / mstd::units::mol},
    {"sup", 1000000.0 * mstd::units::g / mstd::units::mol},
    {"dum", 1.0 * mstd::units::g / mstd::units::mol}
};

#endif   // CONFIG__ATOM_MASS_MAP_HPP
