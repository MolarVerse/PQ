/*****************************************************************************
<GPL_HEADER>

    PQ
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

#include "energyOutput.hpp"

#include <format>    // for format
#include <ostream>   // for basic_ostream, ofstream

#include "constraintSettings.hpp"   // for ConstraintSettings
#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "manostatSettings.hpp"     // for ManostatSettings
#include "physicalData.hpp"         // for PhysicalData
#include "settings.hpp"             // for Settings
#include "thermostatSettings.hpp"   // for ThermostatSettings

using namespace out;
using namespace physicalData;
using namespace settings;

/**
 * @brief Write energy file metadata
 *
 * @param timeStep simulation timestep in fs
 */
void EnergyOutput::writeHeader(double timeStep)
{
    _fp << std::format("# timestep = {} fs\n", timeStep);
}

/**
 * @brief Write the energy output
 *
 * @details
 * - Coulomb and Non-Coulomb energies contain the intra and inter energies.
 * - Bond, Angle, Dihedral and Improper energies are only available if the force
 * field is active.
 * - qm energy is only available if qm is active.
 * - coulomb and non-coulomb energies are only available if mm is active.
 * - volume and density are only available if manostat is active.
 * - nose hoover momentum and friction energies are only available if nose
 * hoover thermostat is active.
 *
 * @param step
 * @param physicalData the physical data of the system
 */
void EnergyOutput::write(size_t step, const PhysicalData &physicalData)
{
    _fp << std::format("{:10d}\t", step);
    _fp << std::format("{:20.12f}\t", physicalData.getTemperature());
    _fp << std::format("{:20.12f}\t", physicalData.getPressure());
    _fp << std::format("{:20.12f}\t", physicalData.getTotalEnergy());

    if (Settings::isQMActivated())
    {
        _fp << std::format("{:20.12f}\t", physicalData.getQMEnergy());
        _fp << std::format("{:20.12f}\t", physicalData.getNumberOfQMAtoms());
    }

    _fp << std::format("{:20.12f}\t", physicalData.getKineticEnergy());
    _fp << std::format("{:20.12f}\t", physicalData.getIntraEnergy());

    if (Settings::isMMActivated())
    {
        _fp << std::format("{:20.12f}\t", physicalData.getCoulombEnergy());
        _fp << std::format("{:20.12f}\t", physicalData.getNonCoulombEnergy());
    }

    if (ForceFieldSettings::isActive())
    {
        _fp << std::format("{:20.12f}\t", physicalData.getBondEnergy());
        _fp << std::format("{:20.12f}\t", physicalData.getAngleEnergy());
        _fp << std::format("{:20.12f}\t", physicalData.getDihedralEnergy());
        _fp << std::format("{:20.12f}\t", physicalData.getImproperEnergy());
    }

    if (Settings::isHybridJobtype())
        _fp << std::format(
            "{:20.12f}\t",
            physicalData.getNumberOfSmoothingMolecules()
        );

    if (ManostatSettings::getManostatType() != ManostatType::NONE)
    {
        _fp << std::format("{:20.12f}\t", physicalData.getVolume());
        _fp << std::format("{:20.12f}\t", physicalData.getDensity());
    }

    if (ThermostatSettings::getThermostatType() == ThermostatType::NOSE_HOOVER)
    {
        _fp << std::format(
            "{:20.12f}\t",
            physicalData.getNoseHooverMomentumEnergy()
        );
        _fp << std::format(
            "{:20.12f}\t",
            physicalData.getNoseHooverFrictionEnergy()
        );
    }

    if (ConstraintSettings::isDistanceConstraintsActivated())
    {
        _fp << std::format(
            "{:20.12f}\t",
            physicalData.getLowerDistanceConstraints()
        );
        _fp << std::format(
            "{:20.12f}\t",
            physicalData.getUpperDistanceConstraints()
        );
    }

    _fp << std::format("{:20.5e}\t", norm(physicalData.getMomentum()));
    _fp << std::format("{:12.5f}\n", physicalData.getLoopTime());

    _fp << std::flush;
}
