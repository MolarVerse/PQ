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
#include "settings.hpp"          // for Settings
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for removeComments, splitString

using namespace input::restartFile;
using namespace simulationBox;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;

using std::make_unique;

/**
 * @brief processes the atom section of the rst file
 *
 * @details this function reads one molecule from the restart file and ends if
 * number of atoms in the molecule is reached. Then the RestartFileReader
 * continues with the next section (possibly the atom section again for the next
 * molecule)
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
    Engine                   &engine
)
{
    auto &simBox = engine.getSimulationBox();

    checkNumberOfLineArguments(lineElements);

    /**********************************
     * find molecule by molecule type *
     *********************************/

    size_t moltype = stoul(lineElements[2]);

    if (0 == moltype)
    {
        processQMAtomLine(lineElements, simBox);
        return;
    }

    std::unique_ptr<MoleculeType> moleculeType;

    try
    {
        // clang-format off
        moleculeType = make_unique<MoleculeType>(simBox.findMoleculeType(moltype));
        // clang-format on
    }
    catch (const RstFileException &e)
    {
        std::cout << e.what() << '\n'
                  << "Error in linenumber " << _lineNumber
                  << " in restart file; Moltype not found\n";

        throw;
    }

    auto molecule = make_unique<Molecule>(moleculeType->getMoltype());

    molecule->setNumberOfAtoms(moleculeType->getNumberOfAtoms());
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
            throw RstFileException(
                std::format(
                    "Error in line {}: Molecule must have {} atoms",
                    _lineNumber,
                    molecule->getNumberOfAtoms()
                )
            );

        processAtomLine(lineElements, simBox, *molecule);

        ++atomCounter;

        if (atomCounter == moleculeType->getNumberOfAtoms())
            break;

        /***********************************************
         * check the next atom line                    *
         * if no atom line is found throw an exception *
         * because if molecule is finished the loop    *
         * should break before                         *
         ***********************************************/

        checkAtomLine(lineElements, *molecule);

        while (lineElements.empty()) checkAtomLine(lineElements, *molecule);

        checkNumberOfLineArguments(lineElements);

        moltype = stoul(lineElements[2]);

        ++_lineNumber;
    }

    simBox.addMolecule(*molecule);
}

/**
 * @brief processes a line of the atom section of the rst file
 *
 * @details the line looks like this:
 * atomTypeName randomEntry MolType x y z vx vy vz fx fy fz
 *
 * @note for backward compatibility the line can also look like this:
 * atomTypeName randomEntry MolType x y z vx vy vz fx fy fz x_old y_old z_old
 * vx_old vy_old vz_old fx_old fy_old fz_old but the old coordinates, velocities
 * and forces are not used and also not read from the file
 *
 * @param lineElements
 * @param simBox
 * @param molecule
 */
void AtomSection::processAtomLine(
    std::vector<std::string> &lineElements,
    SimulationBox            &simBox,
    Molecule                 &molecule
) const
{
    auto atom = std::make_shared<Atom>();

    atom->setAtomTypeName(lineElements[0]);

    setAtomPropertyVectors(lineElements, atom);

    simBox.addAtom(atom);
    molecule.addAtom(atom);

    if (Settings::isQMOnly())
        simBox.addQMAtom(atom);
}

/**
 * @brief adds a single atom with moltype 0 to the simulation box _qmAtoms
 *
 * @details for details how the line looks like see processAtomLine
 *
 * @param lineElements
 * @param simBox
 */
void AtomSection::processQMAtomLine(
    std::vector<std::string> &lineElements,
    SimulationBox            &simBox
)
{
    auto       atom     = std::make_shared<Atom>();
    const auto molecule = make_unique<Molecule>(0);

    molecule->setName("QM");
    molecule->setNumberOfAtoms(1);

    atom->setAtomTypeName(lineElements[0]);
    atom->setName(lineElements[0]);

    setAtomPropertyVectors(lineElements, atom);

    atom->setQMOnly(true);
    molecule->setQMOnly(true);

    molecule->addAtom(atom);

    simBox.addAtom(atom);
    simBox.addQMAtom(atom);
    simBox.addMolecule(*molecule);
}

/**
 * @brief checks if the next line of the rst file exists - if not an
 * exception is thrown
 *
 * @param lineElements
 * @param line
 * @param molecule
 *
 * @throws RstFileException if the next line of the rst
 * file does not exist
 */
void AtomSection::checkAtomLine(
    std::vector<std::string> &lineElements,
    const Molecule           &molecule
)
{
    ++_lineNumber;

    if (std::string line; !getline(*_fp, line))
        throw RstFileException(
            std::format(
                "Error in line {}: Molecule must have {} atoms",
                _lineNumber,
                molecule.getNumberOfAtoms()
            )
        );
    else
    {
        line         = removeComments(line, "#");
        lineElements = splitString(line);
    }
}

/**
 * @brief sets the atom properties from the line elements
 *
 * @param lineElements
 * @param atom
 */
void AtomSection::setAtomPropertyVectors(
    std::vector<std::string> &lineElements,
    std::shared_ptr<Atom>    &atom
) const
{
    try
    {
        const auto x = stringToDouble(lineElements[3]);
        const auto y = stringToDouble(lineElements[4]);
        const auto z = stringToDouble(lineElements[5]);

        atom->setPosition({x, y, z});

        if (lineElements.size() > 6)
        {
            const auto vx = stringToDouble(lineElements[6]);
            const auto vy = stringToDouble(lineElements[7]);
            const auto vz = stringToDouble(lineElements[8]);

            atom->setVelocity({vx, vy, vz});
        }

        if (lineElements.size() > 9)
        {
            const auto fx = stringToDouble(lineElements[9]);
            const auto fy = stringToDouble(lineElements[10]);
            const auto fz = stringToDouble(lineElements[11]);

            atom->setForce({fx, fy, fz});
        }

        if (lineElements.size() > 12)
        {
            const auto oldX = stringToDouble(lineElements[12]);
            const auto oldY = stringToDouble(lineElements[13]);
            const auto oldZ = stringToDouble(lineElements[14]);

            atom->setPositionOld({oldX, oldY, oldZ});
        }

        if (lineElements.size() > 15)
        {
            const auto oldVx = stringToDouble(lineElements[15]);
            const auto oldVy = stringToDouble(lineElements[16]);
            const auto oldVz = stringToDouble(lineElements[17]);

            atom->setVelocityOld({oldVx, oldVy, oldVz});
        }

        if (lineElements.size() > 18)
        {
            const auto oldFx = stringToDouble(lineElements[18]);
            const auto oldFy = stringToDouble(lineElements[19]);
            const auto oldFz = stringToDouble(lineElements[20]);

            atom->setForceOld({oldFx, oldFy, oldFz});
        }
    }
    catch (const std::exception &e)
    {
        throw RstFileException(e.what());
    }
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

    if (lineSize % 3 != 0 || lineSize < 6 || lineSize > 21)
        throw RstFileException(
            std::format(
                "Error in line {}: Atom section must have 6, 9, 12, 15, 18 or "
                "21 elements",
                _lineNumber
            )
        );
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
