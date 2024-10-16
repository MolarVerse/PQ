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

#ifndef _ATOM_MASS_MAP_HPP_

#define _ATOM_MASS_MAP_HPP_

#include <map>
#include <string>

namespace constants
{
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
    const std::map<std::string, double> atomMassMap = {
        {"h", 1.00794},     {"d", 2.014101778}, {"t", 3.0160492675},
        {"he", 4.002602},   {"li", 6.941},      {"be", 9.012182},
        {"b", 10.811},      {"c", 12.0107},     {"n", 14.0067},
        {"o", 15.9994},     {"f", 18.9984032},  {"ne", 20.1797},
        {"na", 22.989770},  {"mg", 24.3050},    {"al", 26.981538},
        {"si", 28.0855},    {"p", 30.973761},   {"s", 32.065},
        {"cl", 35.453},     {"ar", 39.948},     {"k", 39.0983},
        {"ca", 40.078},     {"sc", 44.955910},  {"ti", 47.880},
        {"v", 50.9415},     {"cr", 51.9961},    {"mn", 54.938049},
        {"fe", 55.845},     {"co", 58.933200},  {"ni", 58.6934},
        {"cu", 63.546},     {"zn", 65.399},     {"ga", 69.723},
        {"ge", 72.64},      {"as", 74.92160},   {"se", 78.96},
        {"br", 79.904},     {"kr", 83.798},     {"rb", 85.4678},
        {"sr", 87.62},      {"y", 88.90585},    {"zr", 91.224},
        {"nb", 92.90638},   {"mo", 95.94},      {"tc", 98.9063},
        {"ru", 101.07},     {"rh", 102.9055},   {"pd", 106.42},
        {"ag", 107.8682},   {"cd", 112.411},    {"in", 114.818},
        {"sn", 118.71},     {"sb", 121.76},     {"te", 127.6},
        {"i", 126.90447},   {"xe", 131.293},    {"cs", 132.90546},
        {"ba", 137.327},    {"la", 138.9055},   {"ce", 140.116},
        {"pr", 140.90765},  {"nd", 144.24},     {"pm", 146.9151},
        {"sm", 150.36},     {"lr", 260.1053},   {"eu", 151.964},
        {"gd", 157.25},     {"tb", 158.92534},  {"dy", 162.5},
        {"ho", 164.93032},  {"er", 167.259},    {"tm", 168.93421},
        {"yb", 173.04},     {"lu", 174.967},    {"hf", 178.49},
        {"ta", 180.9479},   {"w", 183.84},      {"re", 186.207},
        {"os", 190.23},     {"ir", 192.217},    {"pt", 195.078},
        {"au", 196.96655},  {"hg", 200.59},     {"tl", 204.3833},
        {"pb", 207.2},      {"bi", 208.98038},  {"po", 208.9824},
        {"at", 209.9871},   {"rn", 222.0176},   {"fr", 223.0197},
        {"ra", 226.0254},   {"ac", 227.0278},   {"th", 232.0381},
        {"pa", 231.03588},  {"u", 238.0289},    {"np", 237.0482},
        {"pu", 244.0642},   {"am", 243.0614},   {"cm", 247.0703},
        {"bk", 247.0703},   {"cf", 251.0796},   {"es", 252.0829},
        {"fm", 257.0951},   {"md", 258.0986},   {"no", 259.1009},
        {"q", 999.00000},   {"x", 999.00000},   {"cav", 1000.00000},
        {"sup", 1000000.0}, {"dum", 1.0}
    };

}   // namespace constants

#endif   // _ATOM_MASS_MAP_HPP_