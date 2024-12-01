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

#include "rstFileOutput.hpp"

#include <format>    // for format
#include <ostream>   // for basic_ostream, operator<<, flush, std
#include <sstream>   // for ostringstream
#include <string>    // for char_traits, operator<<
#include <vector>    // for vector

#include "molecule.hpp"               // for Molecule
#include "noseHooverThermostat.hpp"   // for NoseHooverThermostat
#include "simulationBox.hpp"          // for SimulationBox
#include "thermostatSettings.hpp"     // for ThermostatType
#include "vector3d.hpp"               // for operator<<

using namespace output;
using namespace simulationBox;
using namespace thermostat;
using namespace settings;

/**
 * @brief Write the restart file
 *
 * @param simBox
 * @param step
 */
void RstFileOutput::write(
    SimulationBox    &simBox,
    const Thermostat &thermostat,
    const size_t      step
)
{
    std::ostringstream buffer;

    _fp.close();

    _fp.open(_fileName);

    buffer << "Step " << step << '\n';

    const auto &boxDim = simBox.getBoxDimensions();
    const auto &boxAng = simBox.getBoxAngles();

    buffer << "Box   " << boxDim << "  " << boxAng << '\n';

    if (thermostat.getThermostatType() == ThermostatType::NOSE_HOOVER)
        writeNHChain(thermostat, buffer);

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto nAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < nAtoms; ++i)
        {
            const auto atomName = molecule.getAtomName(i);
            const auto molType  = molecule.getMoltype();
            const auto x        = molecule.getAtomPosition(i)[0];
            const auto y        = molecule.getAtomPosition(i)[1];
            const auto z        = molecule.getAtomPosition(i)[2];
            const auto vx       = molecule.getAtomVelocity(i)[0];
            const auto vy       = molecule.getAtomVelocity(i)[1];
            const auto vz       = molecule.getAtomVelocity(i)[2];
            const auto fx       = molecule.getAtomForce(i)[0];
            const auto fy       = molecule.getAtomForce(i)[1];
            const auto fz       = molecule.getAtomForce(i)[2];

            buffer << std::format("{:<5}\t", atomName);
            buffer << std::format("{:<5}\t", i + 1);
            buffer << std::format("{:<5}\t", molType);

            buffer << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}\t", x, y, z);
            buffer << std::format("{:19.8e}\t{:19.8e}\t{:19.8e}\t", vx, vy, vz);
            buffer << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}", fx, fy, fz);

            buffer << '\n' << std::flush;
        }
    }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief write Nose-Hoover thermostat chi/zeta info to the restart file
 *
 * @param thermostat
 * @param buffer
 */
void RstFileOutput::writeNHChain(
    const Thermostat   &thermostat,
    std::ostringstream &buffer
)
{
    const auto &nh = dynamic_cast<const NoseHooverThermostat &>(thermostat);

    const auto &chi  = nh.getChi();
    const auto &zeta = nh.getZeta();

    for (size_t i = 0; i < chi.size() - 1; ++i)
    {
        buffer
            << "chi "
            << std::format("{:2d}\t{:10.5e}\t{:10.5e}", i + 1, chi[i], zeta[i])
            << '\n';
    }
}