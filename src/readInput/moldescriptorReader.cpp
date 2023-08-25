#include "moldescriptorReader.hpp"

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for MolDescriptorException
#include "forceField.hpp"        // for ForceField
#include "molecule.hpp"          // for Molecule
#include "settings.hpp"          // for Settings
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for removeComments, splitString

#include <boost/algorithm/string/case_conv.hpp>   // for to_lower_copy
#include <boost/iterator/iterator_facade.hpp>     // for operator!=
#include <cstddef>                                // for size_t
#include <format>                                 // for format
#include <string>                                 // for basic_string, string

using namespace std;
using namespace utilities;
using namespace simulationBox;
using namespace readInput;
using namespace engine;
using namespace customException;

/**
 * @brief constructor
 *
 * @details opens moldescritor file pointer
 *
 * @param engine
 *
 * @throw InputFileException if file not found
 */
MoldescriptorReader::MoldescriptorReader(Engine &engine)
    : _filename(engine.getSettings().getMoldescriptorFilename()), _engine(engine)
{
    _fp.open(_filename);

    if (_fp.fail())
        throw InputFileException(format(R"("{}" File not found)", _filename));
}

/**
 * @brief read moldescriptor file
 *
 * @param engine
 *
 * @TODO: for pure QM-MD turn off reading
 */
void readInput::readMolDescriptor(Engine &engine)
{
    MoldescriptorReader reader(engine);
    reader.read();
}

/**
 * @brief read moldescriptor file
 *
 * @throws MolDescriptorException if there is an error in the moldescriptor file
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
        else if (lineElements.size() > 1)
        {
            if (boost::algorithm::to_lower_copy(lineElements[0]) == "water_type")
                _engine.getSimulationBox().setWaterType(stoi(lineElements[1]));
            else if (boost::algorithm::to_lower_copy(lineElements[0]) == "ammonia_type")
                _engine.getSimulationBox().setAmmoniaType(stoi(lineElements[1]));
            else
                processMolecule(lineElements);
        }
        else
            throw MolDescriptorException(format("Error in moldescriptor file at line {}", _lineNumber));
    }
}

/**
 * @brief process molecule
 *
 * @param lineElements
 *
 * @throws MolDescriptorException if there is an error in the moldescriptor file
 */
void MoldescriptorReader::processMolecule(vector<string> &lineElements)
{
    string line;

    if (lineElements.size() < 3)
        throw MolDescriptorException(format("Error in moldescriptor file at line {}", _lineNumber));

    Molecule molecule(lineElements[0]);

    molecule.setNumberOfAtoms(stoul(lineElements[1]));
    molecule.setCharge(stod(lineElements[2]));

    molecule.setMoltype(_engine.getSimulationBox().getMoleculeTypes().size() + 1);

    size_t atomCount = 0;

    while (atomCount < molecule.getNumberOfAtoms())
    {
        if (_fp.eof())
            throw MolDescriptorException(format("Error in moldescriptor file at line {}", _lineNumber));
        getline(_fp, line);
        line         = removeComments(line, "#");
        lineElements = splitString(line);

        ++_lineNumber;

        if (lineElements.empty())
            continue;
        else if ((3 == lineElements.size()) || (4 == lineElements.size()))
        {
            molecule.addAtomName(lineElements[0]);
            molecule.addExternalAtomType(stoul(lineElements[1]));
            molecule.addPartialCharge(stod(lineElements[2]));

            ++atomCount;
        }
        else
            throw MolDescriptorException(format("Error in moldescriptor file at line {}", _lineNumber));

        if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
        {
            if (lineElements.size() != 4)
                throw MolDescriptorException(format("Error in moldescriptor file at line {} - force field noncoulombics is "
                                                    "activated but no global can der Waals parameter given",
                                                    _lineNumber));

            molecule.addExternalGlobalVDWType(stoul(lineElements[3]));
        }
    }

    convertExternalToInternalAtomTypes(molecule);

    _engine.getSimulationBox().getMoleculeTypes().push_back(molecule);
}

/**
 * @brief convert external to internal atom types
 *
 * @details in order to manage if user declares for example only atom type 1 and 3
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