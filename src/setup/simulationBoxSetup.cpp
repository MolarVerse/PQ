#include "simulationBoxSetup.hpp"

#include "atomMassMap.hpp"
#include "atomNumberMap.hpp"
#include "constants.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace setup;
using namespace simulationBox;
using namespace StringUtilities;

/**
 * @brief wrapper to create SetupSimulationBox object and call setup
 *
 * @param engine
 */
void setup::setupSimulationBox(engine::Engine &engine)
{
    SimulationBoxSetup simulationBoxSetup(engine);
    simulationBoxSetup.setup();
}

/**
 * @brief setup simulation box
 *
 */
void SimulationBoxSetup::setup()
{
    setAtomMasses();
    setAtomicNumbers();
    calculateMolMass();
    calculateTotalMass();
    calculateTotalCharge();
    resizeAtomShiftForces();

    checkBoxSettings();
    checkRcCutoff();
}

/**
 * @brief Sets the mass of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void SimulationBoxSetup::setAtomMasses()
{
    const size_t numberOfMolecules = _engine.getSimulationBox().getNumberOfMolecules();

    for (size_t moli = 0; moli < numberOfMolecules; ++moli)
    {
        Molecule    &molecule      = _engine.getSimulationBox().getMolecule(moli);
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto keyword = to_lower_copy(molecule.getAtomName(i));
            if (!constants::atomMassMap.contains(keyword))
                throw customException::MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.addAtomMass(constants::atomMassMap.at(keyword));
        }
    }
}

/**
 * @brief Sets the atomic number of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void SimulationBoxSetup::setAtomicNumbers()
{
    const size_t numberOfMolecules = _engine.getSimulationBox().getNumberOfMolecules();

    for (size_t moli = 0; moli < numberOfMolecules; ++moli)
    {

        Molecule    &molecule      = _engine.getSimulationBox().getMolecule(moli);
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto keyword = to_lower_copy(molecule.getAtomName(i));

            if (!constants::atomNumberMap.contains(keyword))
                throw customException::MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.addAtomicNumber(constants::atomNumberMap.at(keyword));
        }
    }
}

/**
 * @brief calculates the molecular mass of each molecule in the simulation box
 *
 */
void SimulationBoxSetup::calculateMolMass()
{

    const size_t numberOfMolecules = _engine.getSimulationBox().getNumberOfMolecules();

    for (size_t moli = 0; moli < numberOfMolecules; ++moli)
    {
        Molecule    &molecule      = _engine.getSimulationBox().getMolecule(moli);
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        double molMass = 0.0;
        for (size_t i = 0; i < numberOfAtoms; ++i)
            molMass += molecule.getAtomMass(i);

        molecule.setMolMass(molMass);
    }
}

/**
 * @brief Calculates the total mass of the simulation box
 */
void SimulationBoxSetup::calculateTotalMass()
{
    double totalMass = 0.0;

    for (const Molecule &molecule : _engine.getSimulationBox().getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
            totalMass += molecule.getAtomMass(i);
    }

    _engine.getSimulationBox().setTotalMass(totalMass);
}

/**
 * @brief Calculates the total charge of the simulation box
 */
void SimulationBoxSetup::calculateTotalCharge()
{
    double totalCharge = 0.0;

    for (const Molecule &molecule : _engine.getSimulationBox().getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
            totalCharge += molecule.getPartialCharge(i);
    }

    _engine.getSimulationBox().setTotalCharge(totalCharge);
}

/**
 * @brief Checks if the box dimensions and density are set
 *
 * @throw UserInputException if box dimensions and density are not set
 * @throw UserInputExceptionWarning if density and box dimensions are set. Density will be ignored.
 *
 * @note If density is set, box dimensions will be calculated from density.
 * @note If box dimensions are set, density will be calculated from box dimensions.
 */
void SimulationBoxSetup::checkBoxSettings()   // TODO:
{
    auto box     = _engine.getSimulationBox().getBoxDimensions();
    auto density = _engine.getSimulationBox().getDensity();

    if ((density == 0.0) && (box[0] == 0.0) && (box[1] == 0.0) && (box[2] == 0.0))
        throw customException::UserInputException("Box dimensions and density not set");
    else if ((box[0] == 0.0) && (box[1] == 0.0) && (box[2] == 0.0))
    {
        const auto boxDimensions = _engine.getSimulationBox().calculateBoxDimensionsFromDensity();
        _engine.getSimulationBox().setBoxDimensions(boxDimensions);
        _engine.getSimulationBox().setBoxAngles({90.0, 90.0, 90.0});
        _engine.getSimulationBox().setVolume(_engine.getSimulationBox().calculateVolume());
    }
    else if (density == 0.0)
    {
        const auto volume = _engine.getSimulationBox().calculateVolume();
        density = _engine.getSimulationBox().getTotalMass() / volume * constants::_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;
        _engine.getSimulationBox().setVolume(volume);
        _engine.getSimulationBox().setDensity(density);
    }
    else
    {
        const auto volume = _engine.getSimulationBox().calculateVolume();
        density = _engine.getSimulationBox().getTotalMass() / volume * constants::_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;
        _engine.getSimulationBox().setVolume(volume);
        _engine.getSimulationBox().setDensity(density);

        _engine._logOutput->writeDensityWarning();
        _engine._stdoutOutput->writeDensityWarning();
    }

    _engine.getPhysicalData().setVolume(_engine.getSimulationBox().getVolume());
    _engine.getPhysicalData().setDensity(_engine.getSimulationBox().getDensity());
}

/**
 * @brief resizes atomshifvectors to num_atoms
 *
 */
void SimulationBoxSetup::resizeAtomShiftForces()
{
    for (auto &molecule : _engine.getSimulationBox().getMolecules())
        molecule.resizeAtomShiftForces();
}

/**
 * @brief Checks if the cutoff radius is larger than half of the minimal box dimension
 *
 * @throw InputFileException if cutoff radius is larger than half of the minimal box dimension
 */
void SimulationBoxSetup::checkRcCutoff()
{
    if (_engine.getSimulationBox().getRcCutOff() > _engine.getSimulationBox().getMinimalBoxDimension() / 2.0)
        throw customException::InputFileException("Rc cutoff is larger than half of the minimal box dimension of " +
                                                  std::to_string(_engine.getSimulationBox().getMinimalBoxDimension()) +
                                                  " Angstrom.");
}