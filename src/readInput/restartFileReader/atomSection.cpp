#include "atomSection.hpp"

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for RstFileException
#include "molecule.hpp"          // for Molecule
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for removeComments, splitString

#include <cstddef>    // for size_t
#include <format>     // for format
#include <iostream>   // for operator<<, basic_ostream::operator<<
#include <memory>     // for unique_ptr, make_unique
#include <string>     // for string, stod, stoul, getline, char_traits
#include <vector>     // for vector

using namespace readInput::restartFile;

/**
 * @brief processes the atom section of the rst file
 *
 * @details this function reads one molecule from the restart file and ends if number of atoms in the molecule is reached.
 * Then the RestartFileReader continues with the next section (possibly the atom section again for the next molecule)
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

    size_t                                   moltype = stoul(lineElements[2]);
    std::unique_ptr<simulationBox::Molecule> molecule;

    try
    {
        molecule = std::make_unique<simulationBox::Molecule>(engine.getSimulationBox().findMoleculeType(moltype));
    }
    catch (const customException::RstFileException &e)
    {
        std::cout << e.what() << '\n';
        std::cout << "Error in linenumber " << _lineNumber << " in restart file; Moltype not found\n";
        throw;
    }

    size_t atomCounter = 0;

    while (true)
    {
        /********************************************************************************
         * check if molecule type of atom line is the same as the current molecule type *
         ********************************************************************************/

        if (molecule->getMoltype() != moltype)
            throw customException::RstFileException(
                std::format("Error in line {}: Molecule must have {} atoms", _lineNumber, molecule->getNumberOfAtoms()));

        processAtomLine(lineElements, *molecule);

        ++atomCounter;

        if (atomCounter == molecule->getNumberOfAtoms())
            break;

        /* *********************************************
         * check the next atom line                    *
         * if no atom line is found throw an exception *
         * because if molecule is finished the loop    *
         * should break before                         *
         ***********************************************/

        checkAtomLine(lineElements, *molecule);

        while (lineElements.empty())
            checkAtomLine(lineElements, *molecule);

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
 * atomTypeName randomEntry MolType x y z vx vy vz fx fy fz x_old y_old z_old vx_old vy_old vz_old fx_old fy_old fz_old
 * but the old coordinates, velocities and forces are not used and also not read from the file
 *
 * @param lineElements
 * @param molecule
 */
void AtomSection::processAtomLine(std::vector<std::string> &lineElements, simulationBox::Molecule &molecule) const
{
    molecule.addAtomTypeName(lineElements[0]);

    molecule.addAtomPosition({stod(lineElements[3]), stod(lineElements[4]), stod(lineElements[5])});
    molecule.addAtomVelocity({stod(lineElements[6]), stod(lineElements[7]), stod(lineElements[8])});
    molecule.addAtomForce({stod(lineElements[9]), stod(lineElements[10]), stod(lineElements[11])});
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
void AtomSection::checkAtomLine(std::vector<std::string> &lineElements, const simulationBox::Molecule &molecule)
{
    ++_lineNumber;

    if (std::string line; !getline(*_fp, line))
        throw customException::RstFileException(
            std::format("Error in line {}: Molecule must have {} atoms", _lineNumber, molecule.getNumberOfAtoms()));
    else
    {
        line         = utilities::removeComments(line, "#");
        lineElements = utilities::splitString(line);
    }
}

/**
 * @brief checks if the number of elements in the line is correct. The atom section must have 12 or 21 elements.
 *
 * @param lineElements
 *
 * @throws customException::RstFileException if the number of elements in the line is not 12 or 21
 */
void AtomSection::checkNumberOfLineArguments(std::vector<std::string> &lineElements) const
{
    if ((lineElements.size() != 21) && (lineElements.size() != 12))
        throw customException::RstFileException(
            std::format("Error in line {}: Atom section must have 12 or 21 elements", _lineNumber));
}
