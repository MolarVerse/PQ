#include <string>
#include <iostream>

#include "rstFileSection.hpp"

using namespace Setup::RstFileReader;
using namespace std;

string BoxSection::keyword() { return "box"; }
bool BoxSection::isHeader() { return true; }

/**
 * @brief processes the box section of the rst file
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws invalid_argument if the number of elements in the line is not 4 or 7
 */
void BoxSection::process(vector<string> &lineElements, Engine &engine)
{
    if (lineElements.size() != 4 && lineElements.size() != 7)
        throw invalid_argument("Error in line " + to_string(_lineNumber) + ": Box section must have 4 or 7 elements");

    engine._simulationBox._box.setBoxDimensions({stod(lineElements[1]), stod(lineElements[2]), stod(lineElements[3])});

    if (lineElements.size() == 7)
        engine._simulationBox._box.setBoxAngles({stod(lineElements[4]), stod(lineElements[5]), stod(lineElements[6])});
    else
        engine._simulationBox._box.setBoxAngles({90.0, 90.0, 90.0});
}

string NoseHooverSection::keyword() { return "chi"; }
bool NoseHooverSection::isHeader() { return true; }

// TODO: implement this function
void NoseHooverSection::process(vector<string> &, Engine &)
{
    throw invalid_argument("Error in line " + to_string(_lineNumber) + ": Nose-Hoover section not implemented yet");
}

string StepCountSection::keyword() { return "step"; }
bool StepCountSection::isHeader() { return true; }

/**
 * @brief processes the step count section of the rst file
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws invalid_argument if the number of elements in the line is not 2
 */
void StepCountSection::process(vector<string> &lineElements, Engine &engine)
{
    if (lineElements.size() != 2)
        throw invalid_argument("Error in line " + to_string(_lineNumber) + ": Step count section must have 2 elements");

    engine._settings._timings.setStepCount(stoi(lineElements[1]));
}

string AtomSection::keyword() { return nullptr; }
bool AtomSection::isHeader() { return false; }
/**
 * @brief processes the atom section of the rst file
 *
 * @param lineElements all elements of the line
 * @param settings not used in this function
 * @param simulationBox object containing the simulation box
 *
 * @throws invalid_argument if the number of elements in the line is not 21
 */
void AtomSection::process(vector<string> &lineElements, Engine &engine)
{
    if (lineElements.size() != 21)
        throw invalid_argument("Error in line " + to_string(_lineNumber) + ": Atom section must have 21 elements");

    engine._simulationBox.addAtomicProperties(engine._simulationBox._atomtype, lineElements[0]);
    engine._simulationBox.addAtomicProperties(engine._simulationBox._moltype, stoi(lineElements[2]));
    engine._simulationBox.addAtomicProperties(engine._simulationBox._positions, {stod(lineElements[3]), stod(lineElements[4]), stod(lineElements[5])});
    engine._simulationBox.addAtomicProperties(engine._simulationBox._velocities, {stod(lineElements[6]), stod(lineElements[7]), stod(lineElements[8])});
    engine._simulationBox.addAtomicProperties(engine._simulationBox._forces, {stod(lineElements[9]), stod(lineElements[10]), stod(lineElements[11])});
    engine._simulationBox.addAtomicProperties(engine._simulationBox._positionsOld, {stod(lineElements[12]), stod(lineElements[13]), stod(lineElements[14])});
    engine._simulationBox.addAtomicProperties(engine._simulationBox._velocitiesOld, {stod(lineElements[15]), stod(lineElements[16]), stod(lineElements[17])});
    engine._simulationBox.addAtomicProperties(engine._simulationBox._forcesOld, {stod(lineElements[18]), stod(lineElements[19]), stod(lineElements[20])});
}