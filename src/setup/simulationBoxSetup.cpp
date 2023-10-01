/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "simulationBoxSetup.hpp"

#include "atom.hpp"                          // for Atom, simulationBox
#include "atomMassMap.hpp"                   // for atomMassMap
#include "atomNumberMap.hpp"                 // for atomNumberMap
#include "constants/conversionFactors.hpp"   // for _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"                    // for MolDescriptorException
#include "forceFieldSettings.hpp"            // for ForceFieldSettings
#include "logOutput.hpp"                     // for LogOutput
#include "maxwellBoltzmann.hpp"              // for MaxwellBoltzmann
#include "molecule.hpp"                      // for Molecule
#include "physicalData.hpp"                  // for PhysicalData
#include "simulationBox.hpp"                 // for SimulationBox
#include "simulationBoxSettings.hpp"         // for SimulationBoxSettings
#include "stdoutOutput.hpp"                  // for StdoutOutput
#include "stringUtilities.hpp"               // for toLowerCopy, firstLetterToUpperCaseCopy

#include <algorithm>     // for __for_each_fn, for_each
#include <cstddef>       // for size_t
#include <format>        // for format
#include <functional>    // for identity
#include <map>           // for map
#include <numeric>       // for accumulate
#include <string>        // for string, allocator, operator+
#include <string_view>   // for string_view
#include <vector>        // for vector

using namespace setup;
using namespace simulationBox;

/**
 * @brief wrapper to create SetupSimulationBox object and call setup
 *
 * @param engine
 */
void setup::setupSimulationBox(engine::Engine &engine)
{
    engine.getStdoutOutput().writeSetup("simulation box");
    engine.getLogOutput().writeSetup("simulation box");

    SimulationBoxSetup simulationBoxSetup(engine);
    simulationBoxSetup.setup();
}

/**
 * @brief setup simulation box
 *
 * @details
 * 1) set atom masses of each atom in the simulation box
 * 2) set atomic numbers of each atom in the simulation box
 * 3) calculate the molecular mass of each molecule in the simulation box
 * 4) calculate the total mass of the simulation box
 * 5) calculate the total charge of the simulation box
 * 7) check if box dimensions and density are set
 * 8) check if cutoff radius is larger than half of the minimal box dimension
 *
 * @TODO: rewrite doc
 *
 */
void SimulationBoxSetup::setup()
{
    setAtomNames();
    setAtomTypes();
    if (settings::ForceFieldSettings::isActive())
        setExternalVDWTypes();
    setPartialCharges();

    setAtomMasses();
    setAtomicNumbers();
    calculateMolMasses();
    calculateTotalCharge();

    _engine.getSimulationBox().calculateTotalMass();

    checkBoxSettings();
    checkRcCutoff();

    _engine.getSimulationBox().calculateDegreesOfFreedom();
    _engine.getSimulationBox().calculateCenterOfMassMolecules();

    initVelocities();
}

/**
 * @brief set all atomNames in atoms from moleculeTypes
 *
 */
void SimulationBoxSetup::setAtomNames()
{
    auto setAtomNamesOfMolecule = [this](auto &molecule)
    {
        if (molecule.getMoltype() == 0)
            return;

        auto moleculeType = _engine.getSimulationBox().findMoleculeType(molecule.getMoltype());
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
            molecule.getAtom(i).setName(moleculeType.getAtomName(i));
    };

    std::ranges::for_each(_engine.getSimulationBox().getMolecules(), setAtomNamesOfMolecule);

    std::ranges::for_each(_engine.getSimulationBox().getAtoms(),
                          [](auto &atom) { atom->setName(utilities::firstLetterToUpperCaseCopy(atom->getName())); });
}

/**
 * @brief set all external and internal atom types for _atoms from _moleculeTypes
 *
 */
void SimulationBoxSetup::setAtomTypes()
{
    auto setAtomTypesOfMolecule = [this](auto &molecule)
    {
        if (molecule.getMoltype() == 0)
            return;

        auto moleculeType = _engine.getSimulationBox().findMoleculeType(molecule.getMoltype());
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            molecule.getAtom(i).setAtomType(moleculeType.getAtomType(i));
            molecule.getAtom(i).setExternalAtomType(moleculeType.getExternalAtomType(i));
        }
    };

    std::ranges::for_each(_engine.getSimulationBox().getMolecules(), setAtomTypesOfMolecule);
}

/**
 * @brief set all external van der Waals types in atoms from moleculeTypes
 *
 */
void SimulationBoxSetup::setExternalVDWTypes()
{
    auto setExternalVDWTypesOfMolecule = [this](auto &molecule)
    {
        if (molecule.getMoltype() == 0)
            return;

        auto moleculeType = _engine.getSimulationBox().findMoleculeType(molecule.getMoltype());
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
            molecule.getAtom(i).setExternalGlobalVDWType(moleculeType.getExternalGlobalVDWTypes()[i]);
    };

    std::ranges::for_each(_engine.getSimulationBox().getMolecules(), setExternalVDWTypesOfMolecule);
}

/**
 * @brief set all partial charges in atoms from _moleculeTypes
 *
 */
void SimulationBoxSetup::setPartialCharges()
{
    auto setPartialChargesOfMolecule = [this](auto &molecule)
    {
        if (molecule.getMoltype() == 0)
            return;

        auto moleculeType = _engine.getSimulationBox().findMoleculeType(molecule.getMoltype());
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            molecule.getAtom(i).setPartialCharge(moleculeType.getPartialCharges()[i]);
        }
    };

    std::ranges::for_each(_engine.getSimulationBox().getMolecules(), setPartialChargesOfMolecule);
}

/**
 * @brief Sets the mass of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void SimulationBoxSetup::setAtomMasses()
{
    auto setAtomMasses = [](Molecule &molecule)
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            const auto keyword = utilities::toLowerCopy(molecule.getAtomName(i));
            if (!constants::atomMassMap.contains(keyword))
                throw customException::MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.getAtom(i).setMass(constants::atomMassMap.at(keyword));
        }
    };

    std::ranges::for_each(_engine.getSimulationBox().getMolecules(), setAtomMasses);
}

/**
 * @brief Sets the atomic number of each atom in the simulation box
 *
 * @throw MolDescriptorException if atom name is invalid
 */
void SimulationBoxSetup::setAtomicNumbers()
{
    auto setAtomicNumbers = [](Molecule &molecule)
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            const auto keyword = utilities::toLowerCopy(molecule.getAtomName(i));

            if (!constants::atomNumberMap.contains(keyword))
                throw customException::MolDescriptorException("Invalid atom name \"" + keyword + "\"");
            else
                molecule.getAtom(i).setAtomicNumber(constants::atomNumberMap.at(keyword));
        }
    };

    std::ranges::for_each(_engine.getSimulationBox().getMolecules(), setAtomicNumbers);
}

/**
 * @brief calculates the molecular mass of each molecule in the simulation box
 *
 */
void SimulationBoxSetup::calculateMolMasses()
{
    auto calculateMolMasses = [](auto &molecule)
    {
        const auto &masses = molecule.getAtomMasses();
        molecule.setMolMass(std::accumulate(masses.begin(), masses.end(), 0.0));
    };

    std::ranges::for_each(_engine.getSimulationBox().getMolecules(), calculateMolMasses);
}

/**
 * @brief Calculates the total charge of the simulation box
 */
void SimulationBoxSetup::calculateTotalCharge()
{
    double totalCharge = 0.0;

    auto calculateMolecularCharge = [&totalCharge](const Molecule &molecule)
    {
        const auto &charges  = molecule.getPartialCharges();
        totalCharge         += std::accumulate(charges.begin(), charges.end(), 0.0);
    };

    std::ranges::for_each(_engine.getSimulationBox().getMolecules(), calculateMolecularCharge);

    _engine.getSimulationBox().setTotalCharge(totalCharge);
}

/**
 * @brief Checks if the box dimensions and density are set and calculates the missing values
 *
 * @throw UserInputException if box dimensions and density are not set
 * @throw UserInputExceptionWarning if density and box dimensions are set. Density will be ignored.
 *
 * @note If density is set, box dimensions will be calculated from density.
 * @note If box dimensions are set, density will be calculated from box dimensions.
 */
void SimulationBoxSetup::checkBoxSettings()
{
    if (!settings::SimulationBoxSettings::getDensitySet() && !settings::SimulationBoxSettings::getBoxSet())
        throw customException::UserInputException("Box dimensions and density not set");
    else if (!settings::SimulationBoxSettings::getBoxSet())
    {
        const auto boxDimensions = _engine.getSimulationBox().calculateBoxDimensionsFromDensity();
        _engine.getSimulationBox().setBoxDimensions(boxDimensions);
        _engine.getSimulationBox().setVolume(_engine.getSimulationBox().calculateVolume());
    }
    else if (!settings::SimulationBoxSettings::getDensitySet())
    {
        const auto volume = _engine.getSimulationBox().calculateVolume();
        const auto density =
            _engine.getSimulationBox().getTotalMass() / volume * constants::_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;

        _engine.getSimulationBox().setVolume(volume);
        _engine.getSimulationBox().setDensity(density);
    }
    else
    {
        const auto volume = _engine.getSimulationBox().calculateVolume();
        const auto density =
            _engine.getSimulationBox().getTotalMass() / volume * constants::_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_;

        _engine.getSimulationBox().setVolume(volume);
        _engine.getSimulationBox().setDensity(density);

        _engine.getLogOutput().writeDensityWarning();
        _engine.getStdoutOutput().writeDensityWarning();
    }

    _engine.getPhysicalData().setVolume(_engine.getSimulationBox().getVolume());
    _engine.getPhysicalData().setDensity(_engine.getSimulationBox().getDensity());
}

/**
 * @brief Checks if the cutoff radius is larger than half of the minimal box dimension
 *
 * @throw InputFileException if cutoff radius is larger than half of the minimal box dimension
 */
void SimulationBoxSetup::checkRcCutoff()
{
    if (_engine.getSimulationBox().getCoulombRadiusCutOff() > _engine.getSimulationBox().getMinimalBoxDimension() / 2.0)
        throw customException::InputFileException(
            std::format("Rc cutoff is larger than half of the minimal box dimension of {} Angstrom.",
                        _engine.getSimulationBox().getMinimalBoxDimension()));
}

/**
 * @brief Initialize the velocities of the simulation box
 *
 * @details If initializeVelocities is set, the velocities are initialized with a Maxwell-Boltzmann distribution.
 */
void SimulationBoxSetup::initVelocities()
{
    if (settings::SimulationBoxSettings::getInitializeVelocities())
    {
        maxwellBoltzmann::MaxwellBoltzmann maxwellBoltzmann;
        maxwellBoltzmann.initializeVelocities(_engine.getSimulationBox());
    }
}