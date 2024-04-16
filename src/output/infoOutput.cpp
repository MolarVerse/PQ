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

#include "infoOutput.hpp"

#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "manostatSettings.hpp"     // for ManostatSettings
#include "physicalData.hpp"         // for PhysicalData
#include "settings.hpp"             // for Settings
#include "stlVector.hpp"            // for mean, max
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "vector3d.hpp"             // for norm

#include <format>    // for format
#include <ios>       // for ofstream
#include <ostream>   // for operator<<, basic_ostream, char_traits
#include <string>    // for operator<<

using namespace output;

/**
 * @brief write info file
 *
 * @details
 * - Coulomb and Non-Coulomb energies contain the intra and inter energies.
 * - Bond, Angle, Dihedral and Improper energies are only available if the force field is active.
 * - qm energy is only available if qm is active.
 * - coulomb and non-coulomb energies are only available if mm is active.
 * - volume and density are only available if manostat is active.
 * - nose hoover momentum and friction energies are only available if nose hoover thermostat is active.
 *
 * @param simulationTime
 * @param loopTime
 * @param data
 */
void InfoOutput::write(const double simulationTime, const double loopTime, const physicalData::PhysicalData &data)
{
    _fp.close();

    _fp.open(_fileName);

    writeHeader();

    writeLeft(simulationTime, "SIMULATION-TIME", "ps");
    writeRight(data.getTemperature(), "TEMPERATURE", "K");

    writeLeft(data.getPressure(), "PRESSURE", "bar");
    writeRight(data.getTotalEnergy(), "E(TOT)", "kcal/mol");

    if (settings::Settings::isQMActivated())
    {
        writeLeft(data.getQMEnergy(), "E(QM)", "kcal/mol");
        writeRight(data.getNumberOfQMAtoms(), "N(QM-ATOMS)", "-");
    }

    writeLeft(data.getKineticEnergy(), "E(KIN)", "kcal/mol");
    writeRight(data.getIntraEnergy(), "E(INTRA)", "kcal/mol");

    if (settings::Settings::isMMActivated())
    {
        writeLeft(data.getCoulombEnergy(), "E(COUL)", "kcal/mol");
        writeRight(data.getNonCoulombEnergy(), "E(NON-COUL)", "kcal/mol");
    }

    if (settings::ForceFieldSettings::isActive())
    {
        writeLeft(data.getBondEnergy(), "E(BOND)", "kcal/mol");
        writeRight(data.getAngleEnergy(), "E(ANGLE)", "kcal/mol");
        writeLeft(data.getDihedralEnergy(), "E(DIHEDRAL)", "kcal/mol");
        writeRight(data.getImproperEnergy(), "E(IMPROPER)", "kcal/mol");
    }

    if (settings::ManostatSettings::getManostatType() != settings::ManostatType::NONE)
    {
        writeLeft(data.getVolume(), "VOLUME", "A^3");
        writeRight(data.getDensity(), "DENSITY", "g/cm^3");
    }

    if (settings::ThermostatSettings::getThermostatType() == settings::ThermostatType::NOSE_HOOVER)
    {
        writeLeft(data.getNoseHooverMomentumEnergy(), "E(NH-MOMENTUM)", "kcal/mol");
        writeRight(data.getNoseHooverFrictionEnergy(), "E(NH-FRICTION)", "kcal/mol");
    }

    writeLeftScientific(norm(data.getMomentum()), "MOMENTUM", "amuA/fs");
    writeRight(loopTime, "LOOPTIME", "s");

    _fp << std::format("{:-^89}", "") << "\n\n";

    _fp.flush();
}

/**
 * @brief write header of info file
 *
 */
void InfoOutput::writeHeader()
{
    _fp << std::format("{:-^89}", "") << '\n';

    _fp << '|' << std::format("{:^87}", "PQ info file") << '|' << '\n';

    _fp << std::format("{:-^89}", "") << '\n';
}

/**
 * @brief write left column of info file
 *
 * @param value
 * @param name
 * @param unit
 * @param formatter
 * @param precision
 */
void InfoOutput::writeLeft(const double value, const std::string_view &name, const std::string_view &unit)
{
    _fp << std::format("|   {:<15} {:15.5f} {:<8} ", name, value, unit);
}

/**
 * @brief write left column of info file
 *
 * @param value
 * @param name
 * @param unit
 * @param formatter
 * @param precision
 */
void InfoOutput::writeLeftScientific(const double value, const std::string_view &name, const std::string_view &unit)
{
    _fp << std::format("|   {:<15} {:15.1e} {:<8} ", name, value, unit);
}

/**
 * @brief write std::right column of info file
 *
 * @param value
 * @param name
 * @param unit
 * @param formatter
 * @param precision
 */
void InfoOutput::writeRight(const double value, const std::string_view &name, const std::string_view &unit)
{
    _fp << std::format("{:<15} {:15.5f} {:<8}   |\n", name, value, unit);
}
