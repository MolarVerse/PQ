#include <string>
#include <iostream>

#include "rstFileSection.hpp"

using namespace Setup::RstFileReader;

string BoxSection::keyword() { return "box"; }
bool BoxSection::isHeader() { return true; }
void BoxSection::process(vector<string> lineElements, [[maybe_unused]] Settings &settings, SimulationBox &simulationBox)
{
    if (lineElements.size() != 4 || lineElements.size() != 7)
    {
        throw  invalid_argument("Error in line " + to_string(_lineNumber) + ": Box section must have 4 or 7 elements");
    }
}

string NoseHooverSection::keyword() { return "chi"; }
bool NoseHooverSection::isHeader() { return true; }
void NoseHooverSection::process(vector<string> lineElements, Settings &settings, [[maybe_unused]] SimulationBox &simulationBox)
{
    throw invalid_argument("Error in line " + to_string(_lineNumber) + ": Nose-Hoover section not implemented yet");
}

string StepCountSection::keyword() { return "step"; }
bool StepCountSection::isHeader() { return true; }
void StepCountSection::process(vector<string> lineElements, Settings &settings, [[maybe_unused]] SimulationBox &simulationBox)
{
    if (lineElements.size() != 2)
    {
        throw  invalid_argument("Error in line " + to_string(_lineNumber) + ": Step count section must have 2 elements");
    }

    settings.setStepCount(stoi(lineElements[1]));
}

string AtomSection::keyword() { return 0; }
bool AtomSection::isHeader() { return false; }
void AtomSection::process(vector<string> lineElements, [[maybe_unused]] Settings &settings, SimulationBox &simulationBox)
{
    if (lineElements.size() != 21)
    {
        throw  invalid_argument("Error in line " + to_string(_lineNumber) + ": Atom section must have 21 elements");
    }

    // simulationBox.addAtomPositions(stoi(lineElements[0]), stod(lineElements[1]), stod(lineElements[2]), stod(lineElements[3]));
}