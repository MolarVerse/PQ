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

#include "defaults.hpp"
#include "hybridConfigurator.hpp"
#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox

using namespace out;
using namespace defaults;
using namespace configurator;
using namespace molsys;

/**
 * @brief Write the header of a trajectory files
 *
 * @param simulationBox Simulation box containing molecules and atoms.
 */
void TrajectoryOutput::writeHeader(const SimulationBox &simulationBox)
{
    const auto  nAtoms    = simulationBox.getNumberOfAtoms();
    const auto &boxDims   = simulationBox.getBoxDimensions();
    const auto &boxAngles = simulationBox.getBoxAngles();

    _fp << nAtoms << "  " << boxDims << "  " << boxAngles << '\n';
}

/**
 * @brief Write xyz file
 *
 * @param simulationBox Simulation box containing molecules and atoms.
 * @param step
 */
void TrajectoryOutput::writeXyz(const SimulationBox &simulationBox, size_t step)
{
    std::ostringstream buffer;

    writeHeader(simulationBox);
    writeComment(step);

    for (const auto &atom : simulationBox.getAtoms())
    {
        buffer << std::format("{:<5}\t", atom->getName());

        const auto &pos =
            simulationBox.getBox().wrapPositionIntoBox(atom->getPosition());

        buffer << std::format("{:15.8f}\t", pos[0]);
        buffer << std::format("{:15.8f}\t", pos[1]);
        buffer << std::format("{:15.8f}\n", pos[2]);
    }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief Write hybrid center xyz file
 *
 * @param configurator Hybrid configurator containing the inner region center.
 * @param step
 */
void TrajectoryOutput::writeHybridCenterXyz(
    const HybridConfigurator &configurator,
    const size_t              step
)
{
    // one dummy atom is needed to mark the inner region center
    constexpr size_t numberOfCenterAtoms = 1;
    constexpr char   centerAtomName      = INNER_REGION_CENTER_ATOM_NAME;

    // header line
    _fp << numberOfCenterAtoms << '\n';
    writeComment(step);

    std::ostringstream buffer;
    buffer << std::format("{:<5}\t", centerAtomName);

    const auto &pos = configurator.getInnerRegionCenter();

    buffer << std::format("{:15.8f}\t", pos[0]);
    buffer << std::format("{:15.8f}\t", pos[1]);
    buffer << std::format("{:15.8f}\n", pos[2]);

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief Write velocities file
 *
 * @param simulationBox Simulation box containing molecules and atoms.
 * @param step
 */
void TrajectoryOutput::writeVelocities(
    const SimulationBox &simulationBox,
    size_t               step
)
{
    std::ostringstream buffer;

    writeHeader(simulationBox);
    writeComment(step);

    for (const auto &molecule : simulationBox.getMolecules())
    {
        const auto nAtoms = molecule.getNumberOfAtoms();

        for (AtomIndex i{0}; i.get() < nAtoms; ++i)
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
 * @param simulationBox Simulation box containing molecules and atoms.
 * @param step
 */
void TrajectoryOutput::writeForces(
    const SimulationBox &simulationBox,
    size_t               step
)
{
    std::ostringstream buffer;

    writeHeader(simulationBox);
    writeForceComment(step, simulationBox.calculateTotalForce());

    for (const auto &molecule : simulationBox.getMolecules())
    {
        const auto nAtoms = molecule.getNumberOfAtoms();

        for (AtomIndex i{0}; i.get() < nAtoms; ++i)
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
 * @param simulationBox Simulation box containing molecules and atoms.
 * @param step
 */
void TrajectoryOutput::writeCharges(
    const SimulationBox &simulationBox,
    size_t               step
)
{
    std::ostringstream buffer;

    writeHeader(simulationBox);
    writeComment(step);

    for (const auto &atom : simulationBox.getAtoms())
    {
        const auto charge =
            atom->getQMCharge().value_or(atom->getPartialCharge());

        buffer << std::format("{:<5}\t", atom->getName());
        buffer << std::format("{:15.8f}\n", charge);
        buffer << std::flush;
    }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}
