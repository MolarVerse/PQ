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

#include <memory>   // for shared_pointer
#include <string>   // for string
#include <vector>   // for vector

#include "restartFileSection.hpp"   // for RestartFileSection
#include "typeAliases.hpp"          // for strings

#ifdef WITH_TESTS
#include <gtest/gtest_prod.h>   // for FRIEND_TEST

class TestAtomSection_testProcessAtomLine_Test;     // Friend test class
class TestAtomSection_testProcessQMAtomLine_Test;   // Friend test class
#endif

namespace engine
{
    class Engine;   // Forward declaration

}   // namespace engine

namespace simulationBox
{
    class Atom;            // Forward declaration
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
        void processQMAtomLine(pq::strings &lineElements, pq::SimBox &);
        void processAtomLine(pq::strings &, pq::SimBox &, pq::Molecule &) const;

        void checkAtomLine(pq::strings &lineElements, const pq::Molecule &);
        void setAtomPropertyVectors(pq::strings &, pq::SharedAtom &) const;

#ifdef WITH_TESTS
        FRIEND_TEST(::TestAtomSection, testProcessAtomLine);
        FRIEND_TEST(::TestAtomSection, testProcessQMAtomLine);
#endif

       public:
        [[nodiscard]] std::string keyword() override { return ""; }
        [[nodiscard]] bool        isHeader() override { return false; }

        void checkNumberOfLineArguments(pq::strings &) const;
        void process(pq::strings &lineElements, engine::Engine &) override;
    };

}   // namespace input::restartFile

#endif   // _ATOM_SECTION_HPP_
