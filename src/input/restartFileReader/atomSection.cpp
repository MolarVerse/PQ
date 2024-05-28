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

/**
 * @brief processes the atom section of the rst file
 *
 * @details this function reads one molecule from the restart file and ends if number of atoms in
 * the molecule is reached. Then the RestartFileReader continues with the next section (possibly the
 * atom section again for the next molecule)
 *
 * @param lineElements all elements of the line
 * @param engine
 *
 * @throws customException::RstFileException if the molecule type is not found
 * @throws customException::RstFileException if the number of atoms in the molecule is not correct
 */
void AtomSection::process(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    checkNumberOfLineArguments(lineElements);

    /**********************************
     * find molecule by molecule type *
     *********************************/

    size_t moltype = stoul(lineElements[2]);

    if (0 == moltype)
    {
        processQMAtomLine(lineElements, engine.getSimulationBox());
        return;
    }

    std::unique_ptr<simulationBox::MoleculeType> moleculeType;

    try
    {
        moleculeType = std::make_unique<simulationBox::MoleculeType>(
            engine.getSimulationBox().findMoleculeType(moltype)
        );
    }
    catch (const customException::RstFileException &e)
    {
        std::cout << e.what() << '\n';
        std::cout << "Error in linenumber " << _lineNumber
                  << " in restart file; Moltype not found\n";
        throw;
    }

    auto molecule = std::make_unique<simulationBox::Molecule>(moleculeType->getMoltype());

    molecule->setNumberOfAtoms(moleculeType->getNumberOfAtoms());
    molecule->setName(moleculeType->getName());
    molecule->setCharge(moleculeType->getCharge());

    size_t atomCounter = 0;

    while (true)
    {
        /********************************************************************************
         * check if molecule type of atom line is the same as the current molecule type *
         ********************************************************************************/

        if (molecule->getMoltype() != moltype)
            throw customException::RstFileException(std::format(
                "Error in line {}: Molecule must have {} atoms",
                _lineNumber,
                molecule->getNumberOfAtoms()
            ));

        processAtomLine(lineElements, engine.getSimulationBox(), *molecule);

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

    engine.getSimulationBox().addMolecule(*molecule);
}

/**
 * @brief processes a line of the atom section of the rst file
 *
 * @details the line looks like this:
 * atomTypeName randomEntry MolType x y z vx vy vz fx fy fz
 *
 * @note for backward compatibility the line can also look like this:
 * atomTypeName randomEntry MolType x y z vx vy vz fx fy fz x_old y_old z_old vx_old vy_old vz_old
 * fx_old fy_old fz_old but the old coordinates, velocities and forces are not used and also not
 * read from the file
 *
 * @param lineElements
 * @param simBox
 * @param molecule
 */
void AtomSection::processAtomLine(
    std::vector<std::string>     &lineElements,
    simulationBox::SimulationBox &simBox,
    simulationBox::Molecule      &molecule
) const
{
    const auto atom = std::make_shared<simulationBox::Atom>();

    atom->setAtomTypeName(lineElements[0]);

    atom->setPosition({stod(lineElements[3]), stod(lineElements[4]), stod(lineElements[5])});
    atom->setVelocity({stod(lineElements[6]), stod(lineElements[7]), stod(lineElements[8])});
    atom->setForce({stod(lineElements[9]), stod(lineElements[10]), stod(lineElements[11])});

    simBox.addAtom(atom);
    molecule.addAtom(atom);

    if (settings::Settings::isQMOnly())
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
    std::vector<std::string>     &lineElements,
    simulationBox::SimulationBox &simBox
)
{
    auto atom     = std::make_shared<simulationBox::Atom>();
    auto molecule = std::make_unique<simulationBox::Molecule>(0);

    molecule->setName("QM");
    molecule->setNumberOfAtoms(1);

    atom->setAtomTypeName(lineElements[0]);
    atom->setName(lineElements[0]);

    atom->setPosition({stod(lineElements[3]), stod(lineElements[4]), stod(lineElements[5])});
    atom->setVelocity({stod(lineElements[6]), stod(lineElements[7]), stod(lineElements[8])});
    atom->setForce({stod(lineElements[9]), stod(lineElements[10]), stod(lineElements[11])});

    atom->setQMOnly(true);
    molecule->setQMOnly(true);

    molecule->addAtom(atom);

    simBox.addAtom(atom);
    simBox.addQMAtom(atom);
    simBox.addMolecule(*molecule);
}

/**
 * @brief checks if the next line of the rst file exists - if not an exception is thrown
 *
 * @param lineElements
 * @param line
 * @param molecule
 *
 * @throws customException::RstFileException if the next line of the rst file does not exist
 */
void AtomSection::checkAtomLine(
    std::vector<std::string>      &lineElements,
    const simulationBox::Molecule &molecule
)
{
    ++_lineNumber;

    if (std::string line; !getline(*_fp, line))
        throw customException::RstFileException(std::format(
            "Error in line {}: Molecule must have {} atoms",
            _lineNumber,
            molecule.getNumberOfAtoms()
        ));
    else
    {
        line         = utilities::removeComments(line, "#");
        lineElements = utilities::splitString(line);
    }
}

/**
 * @brief checks if the number of elements in the line is correct. The atom section must have 12 or
 * 21 elements.
 *
 * @param lineElements
 *
 * @throws customException::RstFileException if the number of elements in the line is not 12 or 21
 */
void AtomSection::checkNumberOfLineArguments(std::vector<std::string> &lineElements) const
{
    if ((lineElements.size() != 21) && (lineElements.size() != 12))
        throw customException::RstFileException(
            std::format("Error in line {}: Atom section must have 12 or 21 elements", _lineNumber)
        );
}
