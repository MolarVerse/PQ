#include "moldescriptorReader.hpp"

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for customException::MolDescriptorException
#include "fileSettings.hpp"      // for FileSettings
#include "forceFieldClass.hpp"   // for ForceField
#include "molecule.hpp"          // for Molecule
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for removeComments, splitString

#include <cstddef>       // for size_t
#include <format>        // for format
#include <string>        // for basic_string, string
#include <string_view>   // for string_view

using namespace readInput::molDescriptor;

/**
 * @brief constructor
 *
 * @details opens moldescritor file pointer
 *
 * @param engine
 *
 * @throw customException::InputFileException if file not found
 */
MoldescriptorReader::MoldescriptorReader(engine::Engine &engine) : _engine(engine)
{
    _fileName = settings::FileSettings::getMolDescriptorFileName();
    _fp.open(_fileName);
}

/**
 * @brief wrapper to construct MoldescriptorReader and read moldescriptor file
 *
 * @param engine
 *
 * @TODO: for pure QM-MD turn off reading
 */
void readInput::molDescriptor::readMolDescriptor(engine::Engine &engine)
{
    MoldescriptorReader reader(engine);
    reader.read();
}

/**
 * @brief read moldescriptor file
 *
 * @details Processes each line of the moldescriptor file. If a molecule is found the molecule is processed in a separate
 * function. Following keywords are recognized:
 * - water_type <int> - sets the water type
 * - ammonia_type <int> - sets the ammonia type
 * - <molecule_name> <number_of_atoms> <charge> - defines a molecule
 *
 * @throws customException::MolDescriptorException if there is an error in the moldescriptor file
 */
void MoldescriptorReader::read()
{
    std::string line;

    _lineNumber = 0;

    while (getline(_fp, line))
    {
        line              = utilities::removeComments(line, "#");
        auto lineElements = utilities::splitString(line);

        ++_lineNumber;

        if (lineElements.empty())
            continue;
        else if (lineElements.size() > 1)
        {
            if ("water_type" == utilities::toLowerCopy(lineElements[0]))
                _engine.getSimulationBox().setWaterType(std::stoi(lineElements[1]));
            else if ("ammonia_type" == utilities::toLowerCopy(lineElements[0]))
                _engine.getSimulationBox().setAmmoniaType(std::stoi(lineElements[1]));
            else
                processMolecule(lineElements);
        }
        else
            throw customException::MolDescriptorException(std::format("Error in moldescriptor file at line {}", _lineNumber));
    }
}

/**
 * @brief process molecule in moldescriptor file
 *
 * @details Processes the header line of a molecule and then reads the atom lines. The header line has to have following format:
 * <molecule_name> <number_of_atoms> <charge> ...
 * The atom lines have to have following format:
 * <atom_name> <external_atom_type> <partial_charge> [<external_vdw_type>] (external_vdw_type optional if noncoulombics is not
 * activated)
 * After processing the atom lines the external atom types are converted to internal atom types
 *
 * @param lineElements
 *
 * @throws customException::MolDescriptorException if number of arguments of header line is less than 3
 * @throws customException::MolDescriptorException if eof is reached before all atoms of a molecule are read
 * @throws customException::MolDescriptorException if number of arguments of atom line is not 3 or 4
 * @throws customException::MolDescriptorException if noncoulombics is activated but no global van der Waals parameter
 */
void MoldescriptorReader::processMolecule(std::vector<std::string> &lineElements)
{
    if (lineElements.size() < 3)
        throw customException::MolDescriptorException(
            std::format("Not enough arguments in moldescriptor file at line {}", _lineNumber));

    simulationBox::Molecule molecule(lineElements[0]);

    molecule.setNumberOfAtoms(stoul(lineElements[1]));
    molecule.setCharge(stod(lineElements[2]));
    molecule.setMoltype(_engine.getSimulationBox().getMoleculeTypes().size() + 1);

    std::string line;
    size_t      atomCount = 0;

    while (atomCount < molecule.getNumberOfAtoms())
    {
        if (_fp.eof())
            throw customException::MolDescriptorException(
                "Error reading of moldescriptor stopped before last molecule was finished");

        getline(_fp, line);
        line         = utilities::removeComments(line, "#");
        lineElements = utilities::splitString(line);

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
            throw customException::MolDescriptorException(
                std::format("Atom line in moldescriptor file at line {} has to have 3 or 4 elements", _lineNumber));

        if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
        {
            if (lineElements.size() != 4)
                throw customException::MolDescriptorException(
                    std::format("Error in moldescriptor file at line {} - force field noncoulombics is "
                                "activated but no global van der Waals parameter given",
                                _lineNumber));

            molecule.addExternalGlobalVDWType(stoul(lineElements[3]));
        }
    }

    convertExternalToInternalAtomTypes(molecule);

    _engine.getSimulationBox().addMoleculeType(molecule);
}

/**
 * @brief convert external to internal atom types
 *
 * @details In order to manage if user declares for example only atom type 1 and 3 in the moldescriptor file, the internal atom
 * types are the 0 and 1.
 *
 * @param molecule
 */
void MoldescriptorReader::convertExternalToInternalAtomTypes(simulationBox::Molecule &molecule) const
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