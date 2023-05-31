#include "postProcessSetup.hpp"
#include "atomMassMap.hpp"
#include "atomNumberMap.hpp"
#include "exceptions.hpp"
#include "constants.hpp"

#include <map>
#include <string>
#include <boost/algorithm/string.hpp>

using namespace std;

/**
 * @brief Setup post processing
 *
 * @param engine
 */
void postProcessSetup(Engine &engine)
{
    PostProcessSetup postProcessSetup(engine);
    postProcessSetup.setup();
}

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
        Molecule &molecule = _engine.getSimulationBox()._molecules[moli];
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto keyword = boost::algorithm::to_lower_copy(molecule.getAtomName(i));
            if (atomMassMap.find(keyword) == atomMassMap.end())
                throw MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.addMass(atomMassMap.at(keyword));
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

        Molecule &molecule = _engine.getSimulationBox()._molecules[moli];
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
        Molecule &molecule = _engine.getSimulationBox()._molecules[moli];
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        double molMass = 0.0;
        for (size_t i = 0; i < numberOfAtoms; ++i)
            molMass += molecule.getMass(i);

        molecule.setMolMass(molMass);
    }
}

/**
 * @brief Calculates the total mass of the simulation box
 */
void PostProcessSetup::calculateTotalMass()
{
    double totalMass = 0.0;

    for (const Molecule &molecule : _engine.getSimulationBox()._molecules)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
            totalMass += molecule.getMass(i);
    }

    _engine.getSimulationBox()._box.setTotalMass(totalMass);
}

/**
 * @brief Calculates the total charge of the simulation box
 */
void PostProcessSetup::calculateTotalCharge()
{
    double totalCharge = 0.0;

    for (const Molecule &molecule : _engine.getSimulationBox()._molecules)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
            totalCharge += molecule.getPartialCharge(i);
    }

    _engine.getSimulationBox()._box.setTotalCharge(totalCharge);
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
void PostProcessSetup::checkBoxSettings()
{
    auto box = _engine.getSimulationBox()._box.getBoxDimensions();
    auto density = _engine.getSimulationBox()._box.getDensity();

    if ((density == 0.0) && (box[0] == 0.0) && (box[1] == 0.0) && (box[2] == 0.0))
        throw UserInputException("Box dimensions and density not set");
    else if ((box[0] == 0.0) && (box[1] == 0.0) && (box[2] == 0.0))
    {
        const auto boxDimensions = _engine.getSimulationBox()._box.calculateBoxDimensionsFromDensity();
        _engine.getSimulationBox()._box.setBoxDimensions(boxDimensions);
        _engine.getSimulationBox()._box.setBoxAngles({90.0, 90.0, 90.0});
        _engine.getSimulationBox()._box.setVolume(_engine.getSimulationBox()._box.calculateVolume());
    }
    else if (density == 0.0)
    {
        const auto volume = _engine.getSimulationBox()._box.calculateVolume();
        density = _engine.getSimulationBox()._box.getTotalMass() / volume * _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;
        _engine.getSimulationBox()._box.setVolume(volume);
        _engine.getSimulationBox()._box.setDensity(density);
    }
    else
    {
        const auto volume = _engine.getSimulationBox()._box.calculateVolume();
        density = _engine.getSimulationBox()._box.getTotalMass() / volume * _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;
        _engine.getSimulationBox()._box.setVolume(volume);
        _engine.getSimulationBox()._box.setDensity(density);

        _engine._logOutput->writeDensityWarning();
        _engine._stdoutOutput->writeDensityWarning();
    }

    _engine.getPhysicalData().setVolume(_engine.getSimulationBox()._box.getVolume());
    _engine.getPhysicalData().setDensity(_engine.getSimulationBox()._box.getDensity());
}

void PostProcessSetup::resizeAtomShiftForces()
{
    for (auto &molecule : _engine.getSimulationBox()._molecules)
    {
        molecule.resizeAtomShiftForces();
    }
}

void PostProcessSetup::checkRcCutoff()
{
    if (_engine.getSimulationBox().getRcCutOff() > _engine.getSimulationBox()._box.getMinimalBoxDimension() / 2.0)
        throw InputFileException("Rc cutoff is larger than half of the minimal box dimension of " + std::to_string(_engine.getSimulationBox()._box.getMinimalBoxDimension()) + " Angstrom.");
}

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

void PostProcessSetup::setPotential()
{
    if (_engine._potential->getCoulombType() == "guff")
    {
        _engine._potential->_coulombPotential = make_unique<GuffCoulomb>();
    }

    if (_engine._potential->getNonCoulombType() == "guff")
    {
        _engine._potential->_nonCoulombPotential = make_unique<GuffNonCoulomb>();
    }
}