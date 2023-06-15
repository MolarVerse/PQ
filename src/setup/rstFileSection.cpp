#include <string>
#include <iostream>
#include <fstream>

#include "rstFileSection.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace Setup::RstFileReader;
using namespace StringUtilities;
using namespace simulationBox;

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
        throw RstFileException("Error in line " + to_string(_lineNumber) + ": Box section must have 4 or 7 elements");

    engine.getSimulationBox().setBoxDimensions({stod(lineElements[1]), stod(lineElements[2]), stod(lineElements[3])});

    if (lineElements.size() == 7)
        engine.getSimulationBox().setBoxAngles({stod(lineElements[4]), stod(lineElements[5]), stod(lineElements[6])});
    else
        engine.getSimulationBox().setBoxAngles({90.0, 90.0, 90.0});
}

bool NoseHooverSection::isHeader() { return true; }

// TODO: implement this function
void NoseHooverSection::process(vector<string> &, Engine &)
{
    throw RstFileException("Error in line " + to_string(_lineNumber) + ": Nose-Hoover section not implemented yet");
}

bool StepCountSection::isHeader() { return true; }

/**
 * @brief processes the step count section of the rst file
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws RstFileException if the number of elements in the line is not 2
 */
void StepCountSection::process(vector<string> &lineElements, Engine &engine)
{
    if (lineElements.size() != 2)
        throw RstFileException("Error in line " + to_string(_lineNumber) + ": Step count section must have 2 elements");

    engine.getTimings().setStepCount(stoi(lineElements[1]));
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
    string line;
    unique_ptr<Molecule> molecule;

    if (lineElements.size() != 21)
        throw RstFileException("Error in line " + to_string(_lineNumber) + ": Atom section must have 21 elements");

    size_t moltype = stoul(lineElements[2]);

    try
    {
        molecule = make_unique<Molecule>(engine.getSimulationBox().findMoleculeType(moltype));
    }
    catch (const RstFileException &e)
    {
        cout << e.what() << endl;
        cout << "Error in linenumber " + to_string(_lineNumber) + " in restart file" << endl;
        throw;
    }

    size_t atomCounter = 0;

    while (true)
    {
        if (molecule->getMoltype() != moltype)
            throw RstFileException("Error in line " + to_string(_lineNumber) + ": Molecule must have " + to_string(molecule->getNumberOfAtoms()) + " atoms");

        processAtomLine(lineElements, *molecule);

        ++atomCounter;

        if (atomCounter == molecule->getNumberOfAtoms())
            break;

        // TODO: put the next for statements into a function

        checkAtomLine(lineElements, line, *molecule);

        while (lineElements.empty())
        {
            checkAtomLine(lineElements, line, *molecule);
        }

        if ((lineElements.size() != 21) && (lineElements.size() != 12))
            throw RstFileException("Error in line " + to_string(_lineNumber) + ": Atom section must have 12 or 21 elements");

        moltype = stoul(lineElements[2]);
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

    molecule.addAtomPositions({stod(lineElements[3]), stod(lineElements[4]), stod(lineElements[5])});
    molecule.addAtomVelocity({stod(lineElements[6]), stod(lineElements[7]), stod(lineElements[8])});
    molecule.addAtomForces({stod(lineElements[9]), stod(lineElements[10]), stod(lineElements[11])});
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
        throw RstFileException("Error in line " + to_string(_lineNumber) + ": Molecule must have " + to_string(molecule.getNumberOfAtoms()) + " atoms");

    line = removeComments(line, "#");
    // lineElements = splitString(line); TODO: implement all splitString functions via splitString2
    splitString2(line, lineElements);
}