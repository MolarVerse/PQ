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

#include "frame.hpp"

#include <cstddef>

#include "atom.hpp"
#include "molecule.hpp"
#include "noseHooverThermostat.hpp"
#include "simulationBox.hpp"
#include "thermostat.hpp"

namespace out
{
    /**
     * @brief copy box and per-atom data from the simulation box
     *
     * @details atoms are taken in molecule order, which equals the order of
     * the atom list for every box read from a restart file
     *
     * @param simulationBox
     */
    Frame::Frame(const molsys::SimulationBox &simulationBox)
        : box(simulationBox.getBox().clone())
    {
        atoms.reserve(simulationBox.getNumberOfAtoms());

        for (const auto &molecule : simulationBox.getMolecules())
        {
            const auto &moleculeAtoms = molecule.getAtoms();

            for (std::size_t i = 0; i < moleculeAtoms.size(); ++i)
            {
                const auto &atom = *moleculeAtoms[i];

                atoms.push_back({
                    .name            = atom.getName(),
                    .moltype         = molecule.getMoltype(),
                    .indexInMolecule = i,
                    .position        = atom.getPosition(),
                    .velocity        = atom.getVelocity(),
                    .force           = atom.getForce(),
                    .charge =
                        atom.getQMCharge().value_or(atom.getPartialCharge()),
                });
            }
        }
    }

    /**
     * @brief copy box, per-atom data and the Nose-Hoover chain
     *
     * @param simulationBox
     * @param thermostat
     */
    Frame::Frame(
        const molsys::SimulationBox  &simulationBox,
        const thermostat::Thermostat &thermostat
    )
        : Frame(simulationBox)
    {
        if (thermostat.getThermostatType() != ThermostatType::NOSE_HOOVER)
            return;

        const auto &noseHoover =
            dynamic_cast<const thermostat::NoseHooverThermostat &>(thermostat);

        const auto chi  = noseHoover.getChi();
        const auto zeta = noseHoover.getZeta();

        for (std::size_t i = 0; i < chi.size(); ++i)
            noseHooverChain.emplace_back(chi[i], zeta[i]);
    }

    /**
     * @brief norm of the sum of all atomic forces
     *
     * @return double
     */
    double Frame::totalForce() const
    {
        linalg::Vec3D total(0.0);

        for (const auto &atom : atoms) total += atom.force;

        return norm(total);
    }

}   // namespace out
