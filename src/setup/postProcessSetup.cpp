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
    calculateTotalMass();
    calculateTotalCharge();

    checkBoxSettings();
}

/**
 * @brief Sets the mass of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void PostProcessSetup::setAtomMasses()
{
    // TODO: check me!!!!!!!!!!!!
    for (int moli = 0; moli < _engine.getSimulationBox()._molecules.size(); moli++)
    {

        Molecule &molecule = _engine.getSimulationBox()._molecules[moli];
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            auto keyword = boost::algorithm::to_lower_copy(molecule.getAtomName(i));
            if (atomMassMap.find(keyword) == atomMassMap.end())
                throw MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.addMass(atomMassMap.at(keyword));
        }
    }

    cout << _engine.getSimulationBox()._molecules[0].getMass(0) << endl;
}

/**
 * @brief Sets the atomic number of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void PostProcessSetup::setAtomicNumbers()
{
    for (Molecule &molecule : _engine.getSimulationBox()._molecules)
    {
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            auto keyword = boost::algorithm::to_lower_copy(molecule.getAtomName(i));

            if (atomNumberMap.find(keyword) == atomNumberMap.end())
                throw MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.addAtomicNumber(atomNumberMap.at(keyword));
        }
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
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
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
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
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

    if (density == 0.0 && box[0] == 0.0 && box[1] == 0.0 && box[2] == 0.0)
        throw UserInputException("Box dimensions and density not set");
    else if (box[0] == 0.0 && box[1] == 0.0 && box[2] == 0.0)
    {
        auto boxDimensions = _engine.getSimulationBox()._box.calculateBoxDimensionsFromDensity();
        _engine.getSimulationBox()._box.setBoxDimensions(boxDimensions);
        _engine.getSimulationBox()._box.setBoxAngles({90.0, 90.0, 90.0});
    }
    else if (density == 0.0)
    {
        auto volume = _engine.getSimulationBox()._box.calculateVolume();
        density = _engine.getSimulationBox()._box.getTotalMass() / volume * _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;
        _engine.getSimulationBox()._box.setVolume(volume);
        _engine.getSimulationBox()._box.setDensity(density);
    }
    else
    {
        auto volume = _engine.getSimulationBox()._box.calculateVolume();
        density = _engine.getSimulationBox()._box.getTotalMass() / volume * _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;
        _engine.getSimulationBox()._box.setVolume(volume);
        _engine.getSimulationBox()._box.setDensity(density);

        _engine._logOutput->writeDensityWarning();
        _engine._stdoutOutput->writeDensityWarning();
    }
}