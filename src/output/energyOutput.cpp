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
#include <string>    // for operator<<

#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "manostatSettings.hpp"     // for ManostatSettings
#include "physicalData.hpp"         // for PhysicalData
#include "settings.hpp"             // for Settings
#include "stlVector.hpp"            // for mean, max
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "vector3d.hpp"             // for norm

using namespace output;

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
 * @param data
 */
void EnergyOutput::write(
    const size_t                      step,
    const physicalData::PhysicalData &data
)
{
    _fp << std::format("{:10d}\t", step);
    _fp << std::format("{:20.12f}\t", data.getTemperature());
    _fp << std::format("{:20.12f}\t", data.getPressure());
    _fp << std::format("{:20.12f}\t", data.getTotalEnergy());

    if (settings::Settings::isQMActivated())
    {
        _fp << std::format("{:20.12f}\t", data.getQMEnergy());
        _fp << std::format("{:20.12f}\t", data.getNumberOfQMAtoms());
    }

    _fp << std::format("{:20.12f}\t", data.getKineticEnergy());
    _fp << std::format("{:20.12f}\t", data.getIntraEnergy());

    if (settings::Settings::isMMActivated())
    {
        _fp << std::format("{:20.12f}\t", data.getCoulombEnergy());
        _fp << std::format("{:20.12f}\t", data.getNonCoulombEnergy());
    }

    if (settings::ForceFieldSettings::isActive())
    {
        _fp << std::format("{:20.12f}\t", data.getBondEnergy());
        _fp << std::format("{:20.12f}\t", data.getAngleEnergy());
        _fp << std::format("{:20.12f}\t", data.getDihedralEnergy());
        _fp << std::format("{:20.12f}\t", data.getImproperEnergy());
    }

    if (settings::ManostatSettings::getManostatType() !=
        settings::ManostatType::NONE)
    {
        _fp << std::format("{:20.12f}\t", data.getVolume());
        _fp << std::format("{:20.12f}\t", data.getDensity());
    }

    if (settings::ThermostatSettings::getThermostatType() ==
        settings::ThermostatType::NOSE_HOOVER)
    {
        _fp << std::format("{:20.12f}\t", data.getNoseHooverMomentumEnergy());
        _fp << std::format("{:20.12f}\t", data.getNoseHooverFrictionEnergy());
    }

    _fp << std::format("{:20.5e}\t", norm(data.getMomentum()));
    _fp << std::format("{:12.5f}\n", data.getLoopTime());

    _fp << std::flush;
}