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

#ifndef _ATOM_SECTION_HPP_

#define _ATOM_SECTION_HPP_

#include <string>   // for string
#include <vector>   // for vector

#include "restartFileSection.hpp"   // for RestartFileSection

#ifdef WITH_TESTS
#include <gtest/gtest_prod.h>   // for FRIEND_TEST
#endif

class TestAtomSection_testProcessAtomLine_Test;     // Friend test class
class TestAtomSection_testProcessQMAtomLine_Test;   // Friend test class

namespace engine
{
    class Engine;   // Forward declaration
}

namespace simulationBox
{
    class Molecule;        // Forward declaration
    class SimulationBox;   // Forward declaration
}   // namespace simulationBox

namespace input::restartFile
{
    /**
     * @class AtomSection
     *
     * @brief Reads the atom section of a .rst file
     *
     */
    class AtomSection : public RestartFileSection
    {
       private:
        void processAtomLine(std::vector<std::string> &lineElements, simulationBox::SimulationBox &, simulationBox::Molecule &)
            const;
        void processQMAtomLine(std::vector<std::string> &lineElements, simulationBox::SimulationBox &);
        void checkAtomLine(std::vector<std::string> &lineElements, const simulationBox::Molecule &);

#ifdef WITH_TESTS
        FRIEND_TEST(::TestAtomSection, testProcessAtomLine);
        FRIEND_TEST(::TestAtomSection, testProcessQMAtomLine);
#endif

       public:
        [[nodiscard]] std::string keyword() override { return ""; }
        [[nodiscard]] bool        isHeader() override { return false; }
        void checkNumberOfLineArguments(std::vector<std::string> &lineElements
        ) const;
        void process(std::vector<std::string> &lineElements, engine::Engine &)
            override;
    };

}   // namespace input::restartFile

#endif   // _ATOM_SECTION_HPP_
