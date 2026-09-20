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

#include <algorithm>   // for __for_each_fn, for_each
#include <cstddef>     // for size_t
#include <format>      // for format
#include <ostream>     // for basic_ostream, ofstream, operator<<
#include <sstream>     // for ostringstream

#include "molecule.hpp"              // for Molecule
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "simulationBox.hpp"         // for SimulationBox

namespace out
{

    /**
     * @brief write the header of the beads trajectory file
     *
     * @details number of atoms is multiplied by the number of beads - box
     * dimensions and angles are the same for all beads
     *
     * @param simulationBox the simulation box to write the header for
     */
    void RingPolymerTrajectoryOutput::writeHeader(
        const molsys::SimulationBox &simulationBox
    )
    {
        const auto nBeads = settings::RingPolymerSettings::getNumberOfBeads();

        _fp << simulationBox.getNumberOfAtoms() * nBeads << "  ";
        _fp << simulationBox.getBoxDimensions() << "  "
            << simulationBox.getBoxAngles() << '\n';
    }

    /**
     * @brief write the xyz file for all beads
     *
     * @param beads
     * @param step
     */
    void RingPolymerTrajectoryOutput::writeXyz(
        const std::vector<molsys::SimulationBox> &beads,
        size_t                                    step
    )
    {
        std::ostringstream buffer;

        writeHeader(beads[0]);
        writeComment(step);

        const auto nBeads = settings::RingPolymerSettings::getNumberOfBeads();

        for (size_t i = 0; i < nBeads; ++i)
        {
            for (const auto &molecule : beads[i].getMolecules())
            {
                const auto nAtoms = molecule.getNumberOfAtoms();
                for (AtomIndex j{0}; j.get() < nAtoms; ++j)
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
        const std::vector<molsys::SimulationBox> &beads,
        size_t                                    step
    )
    {
        std::ostringstream buffer;

        writeHeader(beads[0]);
        writeComment(step);

        const auto nBeads = settings::RingPolymerSettings::getNumberOfBeads();

        for (size_t i = 0; i < nBeads; ++i)
        {
            for (const auto &molecule : beads[i].getMolecules())
            {
                const auto nAtoms = molecule.getNumberOfAtoms();

                for (AtomIndex j{0}; j.get() < nAtoms; ++j)
                {
                    const auto atomName = molecule.getAtomName(j);
                    const auto velX     = molecule.getAtomVelocity(j)[0];
                    const auto velY     = molecule.getAtomVelocity(j)[1];
                    const auto velZ     = molecule.getAtomVelocity(j)[2];

                    buffer << std::format("{:>5}{}\t", atomName, i + 1);

                    buffer << std::format("{:20.8e}\t", velX);
                    buffer << std::format("{:20.8e}\t", velY);
                    buffer << std::format("{:20.8e}\n", velZ);
                }
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
        const std::vector<molsys::SimulationBox> &beads,
        size_t                                    step
    )
    {
        std::ostringstream buffer;

        writeHeader(beads[0]);

        auto totalForce = 0.0;
        std::ranges::for_each(
            beads,
            [&totalForce](auto &bead)
            { totalForce += bead.calculateTotalForce(); }
        );

        writeForceComment(step, totalForce);

        for (size_t i = 0;
             i < settings::RingPolymerSettings::getNumberOfBeads();
             ++i)
        {
            for (const auto &molecule : beads[i].getMolecules())
            {
                const auto nAtoms = molecule.getNumberOfAtoms();

                for (AtomIndex j{0}; j.get() < nAtoms; ++j)
                {
                    const auto atomName = molecule.getAtomName(j);
                    const auto forceX   = molecule.getAtomForce(j)[0];
                    const auto forceY   = molecule.getAtomForce(j)[1];
                    const auto forceZ   = molecule.getAtomForce(j)[2];

                    buffer << std::format("{:>5}{}\t", atomName, i + 1);

                    buffer << std::format("{:15.8f}\t", forceX);
                    buffer << std::format("{:15.8f}\t", forceY);
                    buffer << std::format("{:15.8f}\n", forceZ);
                }
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
        const std::vector<molsys::SimulationBox> &beads,
        size_t                                    step
    )
    {
        std::ostringstream buffer;

        writeHeader(beads[0]);
        writeComment(step);

        for (size_t i = 0;
             i < settings::RingPolymerSettings::getNumberOfBeads();
             ++i)
        {
            for (const auto &molecule : beads[i].getMolecules())
            {
                for (const auto &atom : molecule.getAtoms())
                {
                    const auto charge =
                        atom->getQMCharge().value_or(atom->getPartialCharge());

                    buffer << std::format("{:>5}{}\t", atom->getName(), i + 1);
                    buffer << std::format("{:15.8f}\n", charge);
                    buffer << std::flush;
                }
            }
        }

        // Write the buffer to the file
        _fp << buffer.str();
        _fp << std::flush;
    }
}   // namespace out
