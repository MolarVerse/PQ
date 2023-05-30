#include <string>
#include <boost/algorithm/string.hpp>
#include <algorithm>

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
MoldescriptorReader::MoldescriptorReader(Engine &engine) : _filename(engine.getSettings().getMoldescriptorFilename()),
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
    MoldescriptorReader reader(engine);
    reader.read();
}

/**
 * @brief read moldescriptor file
 */
void MoldescriptorReader::read()
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
                _engine.getSimulationBox().setWaterType(stoi(lineElements[1]));
            else if (boost::algorithm::to_lower_copy(lineElements[0]) == "ammonia_type")
                _engine.getSimulationBox().setAmmoniaType(stoi(lineElements[1]));
            else
                processMolecule(lineElements);
        }
        else
            throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));
    }
}

/**
 * @brief process molecule
 *
 * @param lineElements
 */
void MoldescriptorReader::processMolecule(vector<string> &lineElements)
{
    string line;

    if (lineElements.size() < 3)
        throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));

    Molecule molecule(lineElements[0]);

    molecule.setNumberOfAtoms(stoi(lineElements[1]));
    molecule.setCharge(stod(lineElements[2]));

    molecule.setMoltype(int(_engine.getSimulationBox()._moleculeTypes.size()) + 1);

    int AtomCount = 0;
    size_t numberOfAtomEntries = 0;

    while (AtomCount < molecule.getNumberOfAtoms())
    {
        getline(_fp, line);
        line = removeComments(line, "#");
        lineElements = splitString(line);

        _lineNumber++;

        numberOfAtomEntries = lineElements.size();

        if (lineElements.empty())
            continue;
        else if (lineElements.size() != numberOfAtomEntries)
            throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));
        else if (lineElements.size() == 3 || lineElements.size() == 4)
        {
            molecule.addAtomName(lineElements[0]);
            molecule.addExternalAtomType(stoi(lineElements[1]));
            molecule.addPartialCharge(stod(lineElements[2]));

            if (lineElements.size() == 4)
                molecule.addGlobalVDWType(stoi(lineElements[3]));

            AtomCount++;
        }
        else
            throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));
    }

    convertExternalToInternalAtomtypes(molecule);

    _engine.getSimulationBox()._moleculeTypes.push_back(molecule);
}

/**
 * @brief convert external to internal atomtypes
 *
 * @details in order to manage if user declares for example only atomtype 1 and 3
 *
 * @param molecule
 */
void MoldescriptorReader::convertExternalToInternalAtomtypes(Molecule &molecule) const
{

    for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
    {
        int externalAtomType = molecule.getExternalAtomType(i);
        molecule.addExternalToInternalAtomTypeElement(externalAtomType, i);
    }

    for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
    {
        int externalAtomType = molecule.getExternalAtomType(i);
        molecule.addAtomType(molecule.getInternalAtomType(externalAtomType));
    }
}