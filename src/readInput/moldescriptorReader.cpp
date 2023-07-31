#include "moldescriptorReader.hpp"

#include "exceptions.hpp"
#include "molecule.hpp"
#include "stringUtilities.hpp"

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <string>

using namespace std;
using namespace StringUtilities;
using namespace simulationBox;
using namespace readInput;
using namespace engine;
using namespace customException;

/**
 * @brief constructor
 *
 * @param engine
 *
 * @throw InputFileException if file not found
 */
MoldescriptorReader::MoldescriptorReader(Engine &engine)
    : _filename(engine.getSettings().getMoldescriptorFilename()), _engine(engine)
{
    _fp.open(_filename);

    if (_fp.fail()) throw InputFileException("\"" + _filename + "\"" + " File not found");
}

/**
 * @brief read moldescriptor file
 *
 * @param engine
 */
void readInput::readMolDescriptor(Engine &engine)
{
    MoldescriptorReader reader(engine);
    reader.read();
}

/**
 * @brief read moldescriptor file
 */
void MoldescriptorReader::read()
{
    string         line;
    vector<string> lineElements;

    _lineNumber = 0;

    while (getline(_fp, line))
    {
        line         = removeComments(line, "#");
        lineElements = splitString(line);

        ++_lineNumber;

        if (lineElements.empty())
            continue;
        else if (lineElements.size() > 1)   // TODO: probably works only for now
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

    if (lineElements.size() < 3) throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));

    Molecule molecule(lineElements[0]);

    molecule.setNumberOfAtoms(stoul(lineElements[1]));
    molecule.setCharge(stod(lineElements[2]));

    molecule.setMoltype(_engine.getSimulationBox().getMoleculeTypes().size() + 1);

    size_t atomCount = 0;

    while (atomCount < molecule.getNumberOfAtoms())
    {
        if (_fp.eof()) throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));
        getline(_fp, line);
        line         = removeComments(line, "#");
        lineElements = splitString(line);

        ++_lineNumber;

        if ((lineElements.size() == 3) || (lineElements.size() == 4))
        {
            molecule.addAtomName(lineElements[0]);
            molecule.addExternalAtomType(stoul(lineElements[1]));
            molecule.addPartialCharge(stod(lineElements[2]));

            if (lineElements.size() == 4) molecule.addGlobalVDWType(stoi(lineElements[3]));

            ++atomCount;
        }
        else
            throw MolDescriptorException("Error in moldescriptor file at line " + to_string(_lineNumber));
    }

    convertExternalToInternalAtomTypes(molecule);

    _engine.getSimulationBox().getMoleculeTypes().push_back(molecule);
}

/**
 * @brief convert external to internal atomtypes
 *
 * @details in order to manage if user declares for example only atomtype 1 and 3
 *
 * @param molecule
 */
void MoldescriptorReader::convertExternalToInternalAtomTypes(Molecule &molecule) const
{
    const size_t numberOfAtoms = molecule.getNumberOfAtoms();

    for (size_t i = 0; i < numberOfAtoms; ++i)
    {
        const size_t externalAtomType = molecule.getExternalAtomType(i);
        molecule.addExternalToInternalAtomTypeElement(externalAtomType, i);
    }

    for (size_t i = 0; i < numberOfAtoms; ++i)
    {
        const size_t externalAtomType = molecule.getExternalAtomType(i);
        molecule.addAtomType(molecule.getInternalAtomType(externalAtomType));
    }
}