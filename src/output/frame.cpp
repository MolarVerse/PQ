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
        : box(simulationBox.getBox().clone()),
          boxDimensions(simulationBox.getBoxDimensions()),
          boxAngles(simulationBox.getBoxAngles()),
          totalForce(simulationBox.calculateTotalForce())
    {
        const auto &allMolecules = simulationBox.getMolecules();
        molecules.reserve(allMolecules.size());

        for (const auto &molecule : allMolecules)
        {
            molecules.push_back(
                {molecule.getMoltype(), molecule.getNumberOfAtoms()}
            );

            for (const auto &atom : molecule.getAtoms())
            {
                names.push_back(atom->getName());
                positions.push_back(atom->getPosition());
                velocities.push_back(atom->getVelocity());
                forces.push_back(atom->getForce());
                charges.push_back(
                    atom->getQMCharge().value_or(atom->getPartialCharge())
                );
            }
        }
    }

    /**
     * @brief copy box, per-atom data and the Nose-Hoover chain state
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

        noseHooverChi  = noseHoover.getChi();
        noseHooverZeta = noseHoover.getZeta();
    }

}   // namespace out
