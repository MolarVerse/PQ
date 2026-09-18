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
#include <vector>    // for vector

#include "molecule.hpp"               // for Molecule
#include "noseHooverThermostat.hpp"   // for NoseHooverThermostat
#include "simulationBox.hpp"          // for SimulationBox
#include "thermostatSettings.hpp"     // for ThermostatType

using namespace out;
using namespace molsys;
using namespace thermostat;
using namespace settings;

namespace
{
    /**
     * @brief write Nose-Hoover thermostat chi/zeta info to the restart file
     *
     * @param thermostat
     * @param buffer
     */
    void writeNHChain(const Thermostat &thermostat, std::ostringstream &buffer)
    {
        const auto &nhChain =
            dynamic_cast<const NoseHooverThermostat &>(thermostat);

        const auto &chi  = nhChain.getChi();
        const auto &zeta = nhChain.getZeta();

        for (size_t i = 0; i < chi.size() - 1; ++i)
        {
            buffer << "chi "
                   << std::format(
                          "{:2d}\t{:10.5e}\t{:10.5e}",
                          i + 1,
                          chi[i],
                          zeta[i]
                      )
                   << '\n';
        }
    }

}   // namespace

/**
 * @brief Write the restart file
 *
 * @param simBox
 * @param thermostat
 * @param step
 */
void RstFileOutput::write(
    SimulationBox    &simBox,
    const Thermostat &thermostat,
    size_t            step
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

        for (AtomIndex i{0}; i.get() < nAtoms; ++i)
        {
            const auto atomName = molecule.getAtomName(i);
            const auto molType  = molecule.getMoltype();
            const auto x        = molecule.getAtomPosition(i)[0];
            const auto y        = molecule.getAtomPosition(i)[1];
            const auto z        = molecule.getAtomPosition(i)[2];
            const auto velX     = molecule.getAtomVelocity(i)[0];
            const auto velY     = molecule.getAtomVelocity(i)[1];
            const auto velZ     = molecule.getAtomVelocity(i)[2];
            const auto forceX   = molecule.getAtomForce(i)[0];
            const auto forceY   = molecule.getAtomForce(i)[1];
            const auto forceZ   = molecule.getAtomForce(i)[2];

            buffer << std::format("{:<5}\t", atomName);
            buffer << std::format("{:<5}\t", i.get() + 1);
            buffer << std::format("{:<5}\t", molType.get());

            buffer << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}\t", x, y, z);
            buffer << std::format(
                "{:19.8e}\t{:19.8e}\t{:19.8e}\t",
                velX,
                velY,
                velZ
            );
            buffer << std::format(
                "{:15.8f}\t{:15.8f}\t{:15.8f}",
                forceX,
                forceY,
                forceZ
            );

            buffer << '\n' << std::flush;
        }
    }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}
