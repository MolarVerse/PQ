#include <string>
#include <iostream>
#include <fstream>

#include "rstFileSection.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace Setup::RstFileReader;
using namespace std;
using namespace StringUtilities;

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
    if (lineElements.size() != 4 && lineElements.size() != 7)
        throw RstFileException("Error in line " + to_string(_lineNumber) + ": Box section must have 4 or 7 elements");

    engine.getSimulationBox()._box.setBoxDimensions({stod(lineElements[1]), stod(lineElements[2]), stod(lineElements[3])});

    if (lineElements.size() == 7)
        engine.getSimulationBox()._box.setBoxAngles({stod(lineElements[4]), stod(lineElements[5]), stod(lineElements[6])});
    else
        engine.getSimulationBox()._box.setBoxAngles({90.0, 90.0, 90.0});
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

    engine._timings.setStepCount(stoi(lineElements[1]));
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

    int moltype = stoi(lineElements[2]);

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

    int atomCounter = 0;

    while (true)
    {
        if (molecule->getMoltype() != moltype)
            throw RstFileException("Error in line " + to_string(_lineNumber) + ": Molecule must have " + to_string(molecule->getNumberOfAtoms()) + " atoms");

        processAtomLine(lineElements, molecule);

        atomCounter++;

        if (atomCounter == molecule->getNumberOfAtoms())
            break;

        // TODO: put the next for statements into a function

        checkAtomLine(lineElements, line, molecule);

        while (lineElements.empty())
        {
            checkAtomLine(lineElements, line, molecule);
        }

        if (lineElements.size() != 21)
            throw RstFileException("Error in line " + to_string(_lineNumber) + ": Atom section must have 21 elements");

        moltype = stoi(lineElements[2]);
    }

    engine.getSimulationBox()._molecules.push_back(*molecule);
}

/**
 * @brief processes a line of the atom section of the rst file
 *
 * @param lineElements
 * @param molecule
 */
void AtomSection::processAtomLine(vector<string> &lineElements, unique_ptr<Molecule> &molecule) const
{
    molecule->addAtomTypeName(lineElements[0]);

    molecule->addAtomPositions({stod(lineElements[3]), stod(lineElements[4]), stod(lineElements[5])});
    molecule->addAtomVelocities({stod(lineElements[6]), stod(lineElements[7]), stod(lineElements[8])});
    molecule->addAtomForce({stod(lineElements[9]), stod(lineElements[10]), stod(lineElements[11])});
    molecule->addAtomPositionOld({stod(lineElements[12]), stod(lineElements[13]), stod(lineElements[14])});
    molecule->addAtomVelocityOld({stod(lineElements[15]), stod(lineElements[16]), stod(lineElements[17])});
    molecule->addAtomForceOld({stod(lineElements[18]), stod(lineElements[19]), stod(lineElements[20])});

    auto velocity = molecule->getAtomVelocities(molecule->getNumberOfAtoms() - 1);
    auto force = molecule->getAtomForce(molecule->getNumberOfAtoms() - 1);

    cout << "velocities: " << velocity[0] << " " << velocity[1] << " " << velocity[2] << endl;
    cout << "velocities2: " << stod(lineElements[6]) << " " << stod(lineElements[7]) << " " << stod(lineElements[8]) << endl;
    cout << "forces: " << force[0] << " " << force[1] << " " << force[2] << endl;
    cout << "forces2: " << stod(lineElements[9]) << " " << stod(lineElements[10]) << " " << stod(lineElements[11]) << endl;
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
void AtomSection::checkAtomLine(vector<string> &lineElements, string &line, unique_ptr<Molecule> &molecule)
{
    _lineNumber++;

    if (!getline(*_fp, line))
        throw RstFileException("Error in line " + to_string(_lineNumber) + ": Molecule must have " + to_string(molecule->getNumberOfAtoms()) + " atoms");

    line = removeComments(line, "#");
    // lineElements = splitString(line); TODO: implement all splitString functions via splitString2
    splitString2(line, lineElements);
}