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

#include "restartFileSection.hpp"   // for RestartFileSection

class TestAtomSection_testProcessAtomLine_Test;     // Friend test class
class TestAtomSection_testProcessQMAtomLine_Test;   // Friend test class

namespace engine
{
    class Engine;   // forward declaration
}   // namespace engine

namespace molsys
{
    class SimulationBox;   // forward declaration
    class Molecule;        // forward declaration
    class Atom;            // forward declaration
}   // namespace molsys

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
        void processQMAtomLine(
            std::vector<std::string> &lineElements,
            molsys::SimulationBox &
        );
        void processAtomLine(
            std::vector<std::string> &,
            molsys::SimulationBox &,
            molsys::Molecule &
        ) const;

        void checkAtomLine(
            std::vector<std::string> &lineElements,
            const molsys::Molecule &
        );
        void setAtomPropertyVectors(
            std::vector<std::string> &,
            std::shared_ptr<molsys::Atom> &
        ) const;

#ifdef WITH_TESTS
        friend class ::TestAtomSection_testProcessAtomLine_Test;
        friend class ::TestAtomSection_testProcessQMAtomLine_Test;
#endif

       public:
        void checkNumberOfLineArguments(std::vector<std::string> &) const;
        void process(
            std::vector<std::string> &lineElements,
            engine::Engine           &engine
        ) override;

        [[nodiscard]] std::string keyword() override;
        [[nodiscard]] bool        isHeader() override;
    };

}   // namespace input::restartFile

#endif   // _ATOM_SECTION_HPP_
