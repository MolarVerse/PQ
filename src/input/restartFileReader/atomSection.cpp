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

#include "atomSection.hpp"

#include <cstddef>    // for size_t
#include <format>     // for format
#include <iostream>   // for operator<<, basic_ostream::operator<<
#include <memory>     // for unique_ptr, make_unique
#include <string>     // for string, stod, stoul, getline, char_traits
#include <vector>     // for vector

#include "atom.hpp"              // for Atom
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for RstFileException
#include "molecule.hpp"          // for Molecule
#include "moleculeType.hpp"      // for MoleculeType
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for removeComments, splitString

namespace input::restartFile
{

    namespace
    {
        /**
         * @brief sets the atom properties from the line elements
         *
         * @param lineElements
         * @param atom
         */
        void setAtomPropertyVectors(
            std::vector<std::string>      &lineElements,
            std::shared_ptr<molsys::Atom> &atom
        )
        {
            try
            {
                const auto x = utilities::stringToFiniteDouble(lineElements[3]);
                const auto y = utilities::stringToFiniteDouble(lineElements[4]);
                const auto z = utilities::stringToFiniteDouble(lineElements[5]);

                atom->setPosition({x, y, z});

                // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
                if (lineElements.size() > 6)
                {
                    const auto velX =
                        utilities::stringToFiniteDouble(lineElements[6]);
                    const auto velY =
                        utilities::stringToFiniteDouble(lineElements[7]);
                    const auto velZ =
                        utilities::stringToFiniteDouble(lineElements[8]);

                    atom->setVelocity({velX, velY, velZ});
                }

                if (lineElements.size() > 9)
                {
                    const auto forceX =
                        utilities::stringToFiniteDouble(lineElements[9]);
                    const auto forceY =
                        utilities::stringToFiniteDouble(lineElements[10]);
                    const auto forceZ =
                        utilities::stringToFiniteDouble(lineElements[11]);

                    atom->setForce({forceX, forceY, forceZ});
                }

                if (lineElements.size() > 12)
                {
                    const auto oldX =
                        utilities::stringToFiniteDouble(lineElements[12]);
                    const auto oldY =
                        utilities::stringToFiniteDouble(lineElements[13]);
                    const auto oldZ =
                        utilities::stringToFiniteDouble(lineElements[14]);

                    atom->setPositionOld({oldX, oldY, oldZ});
                }

                if (lineElements.size() > 15)
                {
                    const auto oldVx =
                        utilities::stringToFiniteDouble(lineElements[15]);
                    const auto oldVy =
                        utilities::stringToFiniteDouble(lineElements[16]);
                    const auto oldVz =
                        utilities::stringToFiniteDouble(lineElements[17]);

                    atom->setVelocityOld({oldVx, oldVy, oldVz});
                }

                if (lineElements.size() > 18)
                {
                    const auto oldFx =
                        utilities::stringToFiniteDouble(lineElements[18]);
                    const auto oldFy =
                        utilities::stringToFiniteDouble(lineElements[19]);
                    const auto oldFz =
                        utilities::stringToFiniteDouble(lineElements[20]);

                    atom->setForceOld({oldFx, oldFy, oldFz});
                }
                // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
            }
            catch (const std::exception &e)
            {
                throw exc::RstFileException(e.what());
            }
        }
    }   // namespace

    /**
     * @brief processes a line of the atom section of the rst file
     *
     * @details the line looks like this:
     * atomTypeName randomEntry MolType x y z vx vy vz fx fy fz
     *
     * @note for backward compatibility the line can also look like this:
     * atomTypeName randomEntry MolType x y z vx vy vz fx fy fz x_old y_old
     * z_old vx_old vy_old vz_old fx_old fy_old fz_old but the old coordinates,
     * velocities and forces are not used and also not read from the file
     *
     * @param lineElements
     * @param simulationBox
     * @param molecule
     */
    void AtomSection::_processAtomLine(
        std::vector<std::string> &lineElements,
        molsys::SimulationBox    &simulationBox,
        molsys::Molecule         &molecule
    )
    {
        auto atom = std::make_shared<molsys::Atom>();

        atom->setAtomTypeName(lineElements[0]);

        setAtomPropertyVectors(lineElements, atom);

        simulationBox.addAtom(atom);
        molecule.addAtom(atom);
    }

    /**
     * @brief adds a single atom with moltype 0 to the simulation box
     *
     * @details for details how the line looks like see processAtomLine
     *
     * @param lineElements
     * @param simulationBox
     */
    void AtomSection::_processQMAtomLine(
        std::vector<std::string> &lineElements,
        molsys::SimulationBox    &simulationBox
    )
    {
        auto       atom     = std::make_shared<molsys::Atom>();
        const auto molecule = std::make_unique<molsys::Molecule>(MolType{0});

        molecule->setName("QM");

        atom->setAtomTypeName(lineElements[0]);
        atom->setName(lineElements[0]);

        setAtomPropertyVectors(lineElements, atom);

        molecule->addAtom(atom);

        simulationBox.addAtom(atom);
        simulationBox.addMolecule(*molecule);
    }

    /**
     * @brief processes the atom section of the rst file
     *
     * @details this function reads one molecule from the restart file and ends
     * if number of atoms in the molecule is reached. Then the RestartFileReader
     * continues with the next section (possibly the atom section again for the
     * next molecule)
     *
     * @param lineElements all elements of the line
     * @param engine
     *
     * @throws RstFileException if the molecule type is not found
     * @throws RstFileException if the number of atoms in the
     * molecule is not correct
     */
    void AtomSection::process(
        std::vector<std::string> &lineElements,
        engine::Engine           &engine
    )
    {
        auto &simulationBox = engine.getSimulationBox();

        checkNumberOfLineArguments(lineElements);

        /**********************************
         * find molecule by molecule type *
         *********************************/

        MolType moltype{stoul(lineElements[2])};

        if (MolType{0} == moltype)
        {
            _processQMAtomLine(lineElements, simulationBox);
            return;
        }

        std::unique_ptr<molsys::MoleculeType> moleculeType;

        try
        {
            // clang-format off
        moleculeType = std::make_unique<molsys::MoleculeType>(simulationBox.findMoleculeType(moltype));
            // clang-format on
        }
        catch (const exc::RstFileException &e)
        {
            std::cout << e.what() << '\n'
                      << "Error in linenumber " << _lineNumber
                      << " in restart file; Moltype not found\n";

            throw;
        }

        auto molecule =
            std::make_unique<molsys::Molecule>(moleculeType->getMoltype());

        molecule->setName(moleculeType->getName());
        molecule->setCharge(moleculeType->getCharge());

        size_t atomCounter = 0;

        while (true)
        {
            /********************************************************************************
             * check if molecule type of atom line is the same as the current
             *molecule type *
             ********************************************************************************/

            if (molecule->getMoltype() != moltype)
            {
                throw exc::RstFileException(
                    std::format(
                        "Error in line {}: Molecule must have {} atoms",
                        _lineNumber,
                        molecule->getNumberOfAtoms()
                    )
                );
            }

            _processAtomLine(lineElements, simulationBox, *molecule);

            ++atomCounter;

            if (atomCounter == moleculeType->getNumberOfAtoms())
                break;

            /***********************************************
             * check the next atom line                    *
             * if no atom line is found throw an exception *
             * because if molecule is finished the loop    *
             * should break before                         *
             ***********************************************/

            _checkAtomLine(lineElements, *molecule);

            while (lineElements.empty())
                _checkAtomLine(lineElements, *molecule);

            checkNumberOfLineArguments(lineElements);

            moltype = MolType{stoul(lineElements[2])};

            ++_lineNumber;
        }

        simulationBox.addMolecule(*molecule);
    }

    /**
     * @brief checks if the next line of the rst file exists - if not an
     * exception is thrown
     *
     * @param lineElements
     * @param molecule
     *
     * @throws RstFileException if the next line of the rst
     * file does not exist
     */
    void AtomSection::_checkAtomLine(
        std::vector<std::string> &lineElements,
        const molsys::Molecule   &molecule
    )
    {
        ++_lineNumber;

        if (std::string line; getline(*_fp, line))
        {
            line         = utilities::removeComments(line, "#");
            lineElements = utilities::splitString(line);
            return;
        }

        throw exc::RstFileException(
            std::format(
                "Error in line {}: Molecule must have {} atoms",
                _lineNumber,
                molecule.getNumberOfAtoms()
            )
        );
    }

    /**
     * @brief checks if the number of elements in the line is correct. The atom
     * section must have 12 or 21 elements.
     *
     * @param lineElements
     *
     * @throws RstFileException if the number of elements in
     * the line is not 12 or 21
     */
    void AtomSection::checkNumberOfLineArguments(
        std::vector<std::string> &lineElements
    ) const
    {
        const auto lineSize = lineElements.size();

        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        if (lineSize % 3 != 0 || lineSize < 6 || lineSize > 21)
        {
            throw exc::RstFileException(
                std::format(
                    "Error in line {}: Atom section must have 6, 9, 12, 15, 18 "
                    "or "
                    "21 elements",
                    _lineNumber
                )
            );
        }
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    }

    /**
     * @brief returns the keyword of the section
     *
     * @return std::string
     */
    std::string AtomSection::keyword() { return ""; }

    /**
     * @brief returns if the section is a header
     *
     * @return bool
     */
    bool AtomSection::isHeader() { return false; }

}   // namespace input::restartFile
