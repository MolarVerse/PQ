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

#include "ringPolymerTrajectoryOutput.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <cstddef>      // for size_t
#include <format>       // for format
#include <ostream>      // for basic_ostream, ofstream, operator<<
#include <sstream>      // for ostringstream

#include "molecule.hpp"              // for Molecule
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "simulationBox.hpp"         // for SimulationBox

using output::RingPolymerTrajectoryOutput;
using namespace settings;
using namespace simulationBox;

/**
 * @brief write the header of the beads trajectory file
 *
 * @details number of atoms is multiplied by the number of beads - box
 * dimensions and angles are the same for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeHeader(const SimulationBox &simBox)
{
    const auto nBeads = RingPolymerSettings::getNumberOfBeads();

    _fp << simBox.getNumberOfAtoms() * nBeads << "  ";
    _fp << simBox.getBoxDimensions() << "  " << simBox.getBoxAngles() << '\n';
}

/**
 * @brief write the xyz file for all beads
 *
 * @param beads
 * @param step
 */
void RingPolymerTrajectoryOutput::writeXyz(
    std::vector<SimulationBox> &beads,
    const size_t                step
)
{
    std::ostringstream buffer;

    writeHeader(beads[0]);
    writeComment(step);

    const auto nBeads = RingPolymerSettings::getNumberOfBeads();

    for (size_t i = 0; i < nBeads; ++i)
        for (const auto &molecule : beads[i].getMolecules())
        {
            const auto nAtoms = molecule.getNumberOfAtoms();
            for (size_t j = 0; j < nAtoms; ++j)
            {
                const auto atomName = molecule.getAtomName(j);
                const auto x        = molecule.getAtomPosition(j)[0];
                const auto y        = molecule.getAtomPosition(j)[1];
                const auto z        = molecule.getAtomPosition(j)[2];

                buffer << std::format("{:>5}{}\t", atomName, i + 1);

                buffer << std::format("{:15.8f}\t", x);
                buffer << std::format("{:15.8f}\t", y);
                buffer << std::format("{:15.8f}\n", z);
            }
        }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief write the velocity file for all beads
 *
 * @param beads
 * @param step
 */
void RingPolymerTrajectoryOutput::writeVelocities(
    std::vector<SimulationBox> &beads,
    const size_t                step
)
{
    std::ostringstream buffer;

    writeHeader(beads[0]);
    writeComment(step);

    const auto nBeads = RingPolymerSettings::getNumberOfBeads();

    for (size_t i = 0; i < nBeads; ++i)
        for (const auto &molecule : beads[i].getMolecules())
        {
            const auto nAtoms = molecule.getNumberOfAtoms();

            for (size_t j = 0; j < nAtoms; ++j)
            {
                const auto atomName = molecule.getAtomName(j);
                const auto vx       = molecule.getAtomVelocity(j)[0];
                const auto vy       = molecule.getAtomVelocity(j)[1];
                const auto vz       = molecule.getAtomVelocity(j)[2];

                buffer << std::format("{:>5}{}\t", atomName, i + 1);

                buffer << std::format("{:20.8e}\t", vx);
                buffer << std::format("{:20.8e}\t", vy);
                buffer << std::format("{:20.8e}\n", vz);
            }
        }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief write the force file for all beads
 *
 * @param beads
 * @param step
 */
void RingPolymerTrajectoryOutput::writeForces(
    std::vector<SimulationBox> &beads,
    const size_t                step
)
{
    std::ostringstream buffer;

    writeHeader(beads[0]);

    auto totalForce = 0.0;
    std::ranges::for_each(
        beads,
        [&totalForce](auto &bead) { totalForce += bead.calculateTotalForce(); }
    );

    writeForceComment(step, totalForce);

    for (size_t i = 0; i < RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
        {
            const auto nAtoms = molecule.getNumberOfAtoms();

            for (size_t j = 0; j < nAtoms; ++j)
            {
                const auto atomName = molecule.getAtomName(j);
                const auto fx       = molecule.getAtomForce(j)[0];
                const auto fy       = molecule.getAtomForce(j)[1];
                const auto fz       = molecule.getAtomForce(j)[2];

                buffer
                    << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                buffer << std::format("{:15.8f}\t", fx);
                buffer << std::format("{:15.8f}\t", fy);
                buffer << std::format("{:15.8f}\n", fz);
            }
        }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}

/**
 * @brief write the charge file for all beads
 *
 * @param beads
 * @param step
 */
void RingPolymerTrajectoryOutput::writeCharges(
    std::vector<SimulationBox> &beads,
    const size_t                step
)
{
    std::ostringstream buffer;

    writeHeader(beads[0]);
    writeComment(step);

    for (size_t i = 0; i < RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
        {
            const auto nAtoms = molecule.getNumberOfAtoms();

            for (size_t j = 0; j < nAtoms; ++j)
            {
                const auto atomName = molecule.getAtomName(j);
                const auto charge   = molecule.getPartialCharge(j);

                buffer << std::format("{:>5}{}\t", atomName, i + 1);
                buffer << std::format("{:15.8f}\n", charge);
                buffer << std::flush;
            }
        }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}
