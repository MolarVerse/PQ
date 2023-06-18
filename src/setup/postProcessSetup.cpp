#include "postProcessSetup.hpp"

#include "atomMassMap.hpp"
#include "atomNumberMap.hpp"
#include "constants.hpp"
#include "exceptions.hpp"

#include <boost/algorithm/string.hpp>
#include <map>
#include <string>

using namespace std;
using namespace simulationBox;
using namespace setup;
using namespace potential;

/**
 * @brief Setup post processing
 *
 * @param engine
 */
void setup::postProcessSetup(Engine &engine)
{
    PostProcessSetup postProcessSetup(engine);
    postProcessSetup.setup();
}

/**
 * @brief wrapper for setup functions
 *
 */
void PostProcessSetup::setup()
{
    setAtomMasses();
    setAtomicNumbers();
    calculateMolMass();
    calculateTotalMass();
    calculateTotalCharge();
    resizeAtomShiftForces();

    checkBoxSettings();

    checkRcCutoff();

    setupCellList();
    setPotential();

    setTimestep();
}

/**
 * @brief Sets the mass of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void PostProcessSetup::setAtomMasses()
{
    const size_t numberOfMolecules = _engine.getSimulationBox().getNumberOfMolecules();

    for (size_t moli = 0; moli < numberOfMolecules; ++moli)
    {
        Molecule    &molecule      = _engine.getSimulationBox().getMolecule(moli);
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto keyword = boost::algorithm::to_lower_copy(molecule.getAtomName(i));
            if (atomMassMap.find(keyword) == atomMassMap.end())
                throw MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.addAtomMass(atomMassMap.at(keyword));
        }
    }
}

/**
 * @brief Sets the atomic number of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void PostProcessSetup::setAtomicNumbers()
{
    const size_t numberOfMolecules = _engine.getSimulationBox().getNumberOfMolecules();

    for (size_t moli = 0; moli < numberOfMolecules; ++moli)
    {

        Molecule    &molecule      = _engine.getSimulationBox().getMolecule(moli);
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto keyword = boost::algorithm::to_lower_copy(molecule.getAtomName(i));

            if (atomNumberMap.find(keyword) == atomNumberMap.end())
                throw MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.addAtomicNumber(atomNumberMap.at(keyword));
        }
    }
}

/**
 * @brief calculates the molecular mass of each molecule in the simulation box
 *
 */
void PostProcessSetup::calculateMolMass()
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
void PostProcessSetup::calculateTotalMass()
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
void PostProcessSetup::calculateTotalCharge()
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
void PostProcessSetup::checkBoxSettings()   // TODO:
{
    auto box     = _engine.getSimulationBox().getBoxDimensions();
    auto density = _engine.getSimulationBox().getDensity();

    if ((density == 0.0) && (box[0] == 0.0) && (box[1] == 0.0) && (box[2] == 0.0))
        throw UserInputException("Box dimensions and density not set");
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
        density           = _engine.getSimulationBox().getTotalMass() / volume * _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;
        _engine.getSimulationBox().setVolume(volume);
        _engine.getSimulationBox().setDensity(density);
    }
    else
    {
        const auto volume = _engine.getSimulationBox().calculateVolume();
        density           = _engine.getSimulationBox().getTotalMass() / volume * _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;
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
void PostProcessSetup::resizeAtomShiftForces()
{
    for (auto &molecule : _engine.getSimulationBox().getMolecules())
        molecule.resizeAtomShiftForces();
}

/**
 * @brief Checks if the cutoff radius is larger than half of the minimal box dimension
 *
 * @throw InputFileException if cutoff radius is larger than half of the minimal box dimension
 */
void PostProcessSetup::checkRcCutoff()
{
    if (_engine.getSimulationBox().getRcCutOff() > _engine.getSimulationBox().getMinimalBoxDimension() / 2.0)
        throw InputFileException("Rc cutoff is larger than half of the minimal box dimension of " +
                                 std::to_string(_engine.getSimulationBox().getMinimalBoxDimension()) + " Angstrom.");
}

/**
 * @brief checks if celllist or brute force potential should be used
 *
 */
void PostProcessSetup::setupCellList()
{
    if (_engine.getCellList().isActivated())
    {
        _engine.getCellList().setup(_engine.getSimulationBox());
        _engine._potential = make_unique<PotentialCellList>();
    }
    else
    {
        _engine._potential = make_unique<PotentialBruteForce>();
    }
}

/**
 * @brief sets all nonbonded potential types
 *
 * @details
 *
 */
void PostProcessSetup::setPotential()
{
    if (_engine._potential->getCoulombType() == "guff") _engine._potential->setCoulombPotential(GuffCoulomb());

    if (_engine._potential->getNonCoulombType() == "guff") _engine._potential->setNonCoulombPotential(GuffNonCoulomb());
}

/**
 * @brief sets timestep in integrator
 *
 */
void PostProcessSetup::setTimestep() { _engine._integrator->setDt(_engine.getTimings().getTimestep()); }