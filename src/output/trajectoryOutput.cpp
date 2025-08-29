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

#include "trajectoryOutput.hpp"

#include <cstddef>   // for size_t
#include <format>    // for format
#include <ostream>   // for ofstream, basic_ostream, operator<<
#include <sstream>   // for ostringstream
#include <string>    // for operator<<
#include <vector>    // for vector

#include "molecule.hpp"                // for Molecule
#include "simulationBox.hpp"           // for SimulationBox
#include "simulationBoxSettings.hpp"   // for SimulationBoxSettings
#include "vector3d.hpp"                // for Vec3D

using namespace output;
using namespace settings;
using namespace simulationBox;

/**
 * @brief Write the header of a trajectory files
 *
 * @param simBox
 */
void TrajectoryOutput::writeHeader(const SimulationBox &simBox)
{
    const auto  nAtoms    = simBox.getNumberOfAtoms();
    const auto &boxDims   = simBox.getBoxDimensions();
    const auto &boxAngles = simBox.getBoxAngles();

    _fp << nAtoms << "  " << boxDims << "  " << boxAngles << '\n';
}

/**
 * @brief Write xyz file
 *
 * @param simBox
 */
void TrajectoryOutput::writeXyz(SimulationBox &simBox)
{
    std::ostringstream buffer;

    writeHeader(simBox);
    buffer << '\n';

    for (const auto &atom : simBox.getAtoms())
    {
        buffer << std::format("{:<5}\t", atom->getName());

        auto pos = atom->getPosition();

        if (SimulationBoxSettings::isBoxTriclinic())
            pos = simBox.getBox().wrapPositionsIntoBox(pos);

        buffer << std::format("{:15.8f}\t", pos[0]);
        buffer << std::format("{:15.8f}\t", pos[1]);
        buffer << std::format("{:15.8f}\n", pos[2]);
    }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief Write velocities file
 *
 * @param simBox
 */
void TrajectoryOutput::writeVelocities(SimulationBox &simBox)
{
    std::ostringstream buffer;

    writeHeader(simBox);
    buffer << '\n';

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto nAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < nAtoms; ++i)
        {
            buffer << std::format("{:<5}\t", molecule.getAtomName(i));

            const auto &vel = molecule.getAtomVelocity(i);

            buffer << std::format("{:20.8e}\t", vel[0]);
            buffer << std::format("{:20.8e}\t", vel[1]);
            buffer << std::format("{:20.8e}\n", vel[2]);
        }
    }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief Write forces file
 *
 * @param simBox
 */
void TrajectoryOutput::writeForces(SimulationBox &simBox)
{
    std::ostringstream buffer;

    writeHeader(simBox);
    buffer << std::format(
        "# Total force = {:.5e} kcal/mol/Angstrom\n",
        simBox.calculateTotalForce()
    );

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto nAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < nAtoms; ++i)
        {
            buffer << std::format("{:<5}\t", molecule.getAtomName(i));

            const auto &force = molecule.getAtomForce(i);

            buffer << std::format("{:15.8f}\t", force[0]);
            buffer << std::format("{:15.8f}\t", force[1]);
            buffer << std::format("{:15.8f}\n", force[2]);
        }
    }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief Write charges file
 *
 * @param simBox
 */
void TrajectoryOutput::writeCharges(SimulationBox &simBox)
{
    std::ostringstream buffer;

    writeHeader(simBox);
    buffer << '\n';

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto nAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < nAtoms; ++i)
        {
            buffer << std::format("{:<5}\t", molecule.getAtomName(i));
            buffer << std::format("{:15.8f}\n", molecule.getPartialCharge(i));
            buffer << std::flush;
        }
    }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}