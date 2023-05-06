#include <string>
#include <boost/algorithm/string.hpp>

#include "moldescriptorReader.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"
#include "molecule.hpp"

using namespace std;
using namespace StringUtilities;

/**
 * @brief constructor
 *
 * @param engine
 *
 * @throw InputFileException if file not found
 */
MolDescriptorReader::MolDescriptorReader(Engine &engine) : _filename(engine._settings.getMoldescriptorFilename()),
                                                           _engine(engine)
{
    _fp.open(_filename);

    if (_fp.fail())
        throw InputFileException("\"" + _filename + "\"" + " File not found");
}

/**
 * @brief read moldescriptor file
 *
 * @param engine
 */
void readMolDescriptor(Engine &engine)
{
    MolDescriptorReader reader(engine);
    reader.read();
}

/**
 * @brief read moldescriptor file
 */
void MolDescriptorReader::read()
{
    string line;
    vector<string> lineElements;

    _lineNumber = 0;

    while (getline(_fp, line))
    {
        line = removeComments(line, "#");
        lineElements = splitString(line);

        _lineNumber++;

        if (lineElements.empty())
            continue;
        else if (lineElements.size() > 1) // TODO: probably works only for now
        {
            if (boost::algorithm::to_lower_copy(lineElements[0]) == "water_type")
                _engine._simulationBox.setWaterType(stoi(lineElements[1]));
            else if (boost::algorithm::to_lower_copy(lineElements[0]) == "ammonia_type")
                _engine._simulationBox.setAmmoniaType(stoi(lineElements[1]));
            else
                processMolecule(lineElements);
        }
        else
            throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));
    }
}

void MolDescriptorReader::processMolecule(vector<string> &lineElements)
{
    string line;

    if (lineElements.size() < 3)
        throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));

    Molecule molecule(lineElements[0]);

    molecule.setNumberOfAtoms(stoi(lineElements[1]));
    molecule.setCharge(stod(lineElements[2]));
}