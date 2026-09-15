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

#ifndef _ATOM_NUMBER_MAP_HPP_

#define _ATOM_NUMBER_MAP_HPP_

#include <map>
#include <string>

#include "strongTypes.hpp"

namespace constants
{
    /**
     * @brief Map of atom names to atomic numbers
     *
     * @note special atom names are:
     *
     * d deuterium
     * t tritium
     *
     * q with atomic number 999
     * x with atomic number 999
     * cav with atomic number 1000
     * sup with atomic number 1000000
     * dum with atomic number 1
     */
    const std::map<std::string, AtomNumber> atomNumberMap = {
        {"h", AtomNumber{1}},         {"d", AtomNumber{1}},
        {"t", AtomNumber{1}},         {"he", AtomNumber{2}},
        {"li", AtomNumber{3}},        {"be", AtomNumber{4}},
        {"b", AtomNumber{5}},         {"c", AtomNumber{6}},
        {"n", AtomNumber{7}},         {"o", AtomNumber{8}},
        {"f", AtomNumber{9}},         {"ne", AtomNumber{10}},
        {"na", AtomNumber{11}},       {"mg", AtomNumber{12}},
        {"al", AtomNumber{13}},       {"si", AtomNumber{14}},
        {"p", AtomNumber{15}},        {"s", AtomNumber{16}},
        {"cl", AtomNumber{17}},       {"ar", AtomNumber{18}},
        {"k", AtomNumber{19}},        {"ca", AtomNumber{20}},
        {"sc", AtomNumber{21}},       {"ti", AtomNumber{22}},
        {"v", AtomNumber{23}},        {"cr", AtomNumber{24}},
        {"mn", AtomNumber{25}},       {"fe", AtomNumber{26}},
        {"co", AtomNumber{27}},       {"ni", AtomNumber{28}},
        {"cu", AtomNumber{29}},       {"zn", AtomNumber{30}},
        {"ga", AtomNumber{31}},       {"ge", AtomNumber{32}},
        {"as", AtomNumber{33}},       {"se", AtomNumber{34}},
        {"br", AtomNumber{35}},       {"kr", AtomNumber{36}},
        {"rb", AtomNumber{37}},       {"sr", AtomNumber{38}},
        {"y", AtomNumber{39}},        {"zr", AtomNumber{40}},
        {"nb", AtomNumber{41}},       {"mo", AtomNumber{42}},
        {"tc", AtomNumber{43}},       {"ru", AtomNumber{44}},
        {"rh", AtomNumber{45}},       {"pd", AtomNumber{46}},
        {"ag", AtomNumber{47}},       {"cd", AtomNumber{48}},
        {"in", AtomNumber{49}},       {"sn", AtomNumber{50}},
        {"sb", AtomNumber{51}},       {"te", AtomNumber{52}},
        {"i", AtomNumber{53}},        {"xe", AtomNumber{54}},
        {"cs", AtomNumber{55}},       {"ba", AtomNumber{56}},
        {"la", AtomNumber{57}},       {"ce", AtomNumber{58}},
        {"pr", AtomNumber{59}},       {"nd", AtomNumber{60}},
        {"pm", AtomNumber{61}},       {"sm", AtomNumber{62}},
        {"eu", AtomNumber{63}},       {"gd", AtomNumber{64}},
        {"tb", AtomNumber{65}},       {"dy", AtomNumber{66}},
        {"ho", AtomNumber{67}},       {"er", AtomNumber{68}},
        {"tm", AtomNumber{69}},       {"yb", AtomNumber{70}},
        {"lu", AtomNumber{71}},       {"hf", AtomNumber{72}},
        {"ta", AtomNumber{73}},       {"w", AtomNumber{74}},
        {"re", AtomNumber{75}},       {"os", AtomNumber{76}},
        {"ir", AtomNumber{77}},       {"pt", AtomNumber{78}},
        {"au", AtomNumber{79}},       {"hg", AtomNumber{80}},
        {"tl", AtomNumber{81}},       {"pb", AtomNumber{82}},
        {"bi", AtomNumber{83}},       {"po", AtomNumber{84}},
        {"at", AtomNumber{85}},       {"rn", AtomNumber{86}},
        {"fr", AtomNumber{87}},       {"ra", AtomNumber{88}},
        {"ac", AtomNumber{89}},       {"th", AtomNumber{90}},
        {"pa", AtomNumber{91}},       {"u", AtomNumber{92}},
        {"np", AtomNumber{93}},       {"pu", AtomNumber{94}},
        {"am", AtomNumber{95}},       {"cm", AtomNumber{96}},
        {"bk", AtomNumber{97}},       {"cf", AtomNumber{98}},
        {"es", AtomNumber{99}},       {"fm", AtomNumber{100}},
        {"md", AtomNumber{101}},      {"no", AtomNumber{102}},
        {"lr", AtomNumber{103}},      {"q", AtomNumber{999}},
        {"x", AtomNumber{999}},       {"cav", AtomNumber{1000}},
        {"sup", AtomNumber{1000000}}, {"dum", AtomNumber{1}}
    };

}   // namespace constants

#endif   // _ATOM_NUMBER_MAP_HPP_
