#include "rstFileSection.hpp"

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for RstFileException, customException
#include "molecule.hpp"          // for Molecule
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for removeComments, splitString, utilities
#include "timings.hpp"           // for Timings

#include <cstddef>    // for size_t
#include <format>     // for format
#include <fstream>    // for operator<<, basic_ostream::operator<<
#include <iostream>   // for cout
#include <memory>     // for unique_ptr, make_unique
#include <string>     // for stod, string, stoul, getline, stoi

using namespace std;
using namespace utilities;
using namespace simulationBox;
using namespace readInput;
using namespace engine;
using namespace customException;

bool BoxSection::isHeader() { return true; }

/**
 * @brief processes the box section of the rst file
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws RstFileException if the number of elements in the line is not 4 or 7
 */
void BoxSection::process(vector<string> &lineElements, Engine &engine)
{
    if ((lineElements.size() != 4) && (lineElements.size() != 7))
        throw RstFileException(format("Error in line {}: Box section must have 4 or 7 elements", _lineNumber));

    engine.getSimulationBox().setBoxDimensions({stod(lineElements[1]), stod(lineElements[2]), stod(lineElements[3])});

    if (7 == lineElements.size())
        engine.getSimulationBox().setBoxAngles({stod(lineElements[4]), stod(lineElements[5]), stod(lineElements[6])});
    else
        engine.getSimulationBox().setBoxAngles({90.0, 90.0, 90.0});
}

bool NoseHooverSection::isHeader() { return true; }

void NoseHooverSection::process(vector<string> &, Engine &)
{
    throw RstFileException(format("Error in line {}: Nose-Hoover section not implemented yet", _lineNumber));
}

bool StepCountSection::isHeader() { return true; }

/**
 * @brief processes the step count section of the rst file
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws RstFileException if the number of elements in the line is not 2
 * @throws RstFileException if the step count is negative
 */
void StepCountSection::process(vector<string> &lineElements, Engine &engine)
{
    if (lineElements.size() != 2)
        throw RstFileException(format("Error in line {}: Step count section must have 2 elements", _lineNumber));

    auto stepCount = stoi(lineElements[1]);

    if (stepCount < 0)
        throw RstFileException(format("Error in line {}: Step count must be positive", _lineNumber));

    engine.getTimings().setStepCount(size_t(stepCount));
}

bool AtomSection::isHeader() { return false; }

/**
 * @brief processes the atom section of the rst file
 *
 * @param lineElements all elements of the line
 * @param engine
 *
 * @throws RstFileException if the number of elements in the line is not 21
 * @throws RstFileException if the molecule type is not found
 * @throws RstFileException if the number of atoms in the molecule is not correct
 */
void AtomSection::process(vector<string> &lineElements, Engine &engine)
{
    string               line;
    unique_ptr<Molecule> molecule;

    if (lineElements.size() != 21)
        throw RstFileException(format("Error in line {}: Atom section must have 21 elements", _lineNumber));

    size_t moltype = stoul(lineElements[2]);

    try
    {
        molecule = make_unique<Molecule>(engine.getSimulationBox().findMoleculeType(moltype));
    }
    catch (const RstFileException &e)
    {
        cout << e.what() << '\n';
        cout << "Error in linenumber " << _lineNumber << " in restart file; Moltype not found\n";
        throw;
    }

    size_t atomCounter = 0;

    while (true)
    {
        if (molecule->getMoltype() != moltype)
            throw RstFileException(
                format("Error in line {}: Molecule must have {} atoms", _lineNumber, molecule->getNumberOfAtoms()));

        processAtomLine(lineElements, *molecule);

        ++atomCounter;

        if (atomCounter == molecule->getNumberOfAtoms())
            break;

        checkAtomLine(lineElements, line, *molecule);

        while (lineElements.empty())
        {
            checkAtomLine(lineElements, line, *molecule);
        }

        if ((lineElements.size() != 21) && (lineElements.size() != 12))
            throw RstFileException(format("Error in line {}: Atom section must have 12 or 21 elements", _lineNumber));

        moltype = stoul(lineElements[2]);

        ++_lineNumber;
    }

    engine.getSimulationBox().addMolecule(*molecule);
}

/**
 * @brief processes a line of the atom section of the rst file
 *
 * @param lineElements
 * @param molecule
 */
void AtomSection::processAtomLine(vector<string> &lineElements, Molecule &molecule) const
{
    molecule.addAtomTypeName(lineElements[0]);

    molecule.addAtomPosition({stod(lineElements[3]), stod(lineElements[4]), stod(lineElements[5])});
    molecule.addAtomVelocity({stod(lineElements[6]), stod(lineElements[7]), stod(lineElements[8])});
    molecule.addAtomForce({stod(lineElements[9]), stod(lineElements[10]), stod(lineElements[11])});
}

/**
 * @brief checks if the line of the atom section of the rst file is correct
 *
 * @param lineElements
 * @param line
 * @param molecule
 *
 * @throws RstFileException if the number of elements in the line is not 21
 */
void AtomSection::checkAtomLine(vector<string> &lineElements, string &line, const Molecule &molecule)
{
    ++_lineNumber;

    if (!getline(*_fp, line))
        throw RstFileException(format("Error in line {}: Molecule must have {} atoms", _lineNumber, molecule.getNumberOfAtoms()));

    line         = removeComments(line, "#");
    lineElements = splitString(line);
}