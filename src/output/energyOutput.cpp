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

#include <format>
#include <ostream>

#include "constraintSettings.hpp"
#include "forceFieldSettings.hpp"
#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "physicalData.hpp"
#include "settings.hpp"
#include "thermostatSettings.hpp"

namespace out
{

    /**
     * @brief Write energy file metadata
     *
     * @param timeStep simulation timestep in fs
     */
    void EnergyOutput::writeHeader(double timeStep)
    {
        _getFilePointer() << std::format("# timestep = {} fs\n", timeStep);
    }

    /**
     * @brief Write the energy output
     *
     * @details
     * - Coulomb and Non-Coulomb energies contain the intra and inter energies.
     * - Bond, Angle, Dihedral and Improper energies are only available if the
     * force field is active.
     * - qm energy is only available if qm is active.
     * - coulomb and non-coulomb energies are only available if mm is active.
     * - volume and density are only available if manostat is active.
     * - nose hoover momentum and friction energies are only available if nose
     * hoover thermostat is active.
     *
     * @param step
     * @param physicalData the physical data of the system
     * @param settings The simulation settings containing relevant configuration
     * for output.
     */
    void EnergyOutput::write(
        size_t                            step,
        const physicalData::PhysicalData &physicalData,
        const Settings                   &settings
    )
    {
        _getFilePointer() << std::format("{:10d}\t", step);
        _getFilePointer(
        ) << std::format("{:20.12f}\t", physicalData.getTemperature());
        _getFilePointer(
        ) << std::format("{:20.12f}\t", physicalData.getPressure());
        _getFilePointer(
        ) << std::format("{:20.12f}\t", physicalData.getTotalEnergy());

        if (settings::GeneralSettings::isQMActivated())
        {
            _getFilePointer()
                << std::format("{:20.12f}\t", physicalData.getQMEnergy());
            _getFilePointer(
            ) << std::format("{:20.12f}\t", physicalData.getNumberOfQMAtoms());
        }

        _getFilePointer(
        ) << std::format("{:20.12f}\t", physicalData.getKineticEnergy());
        _getFilePointer(
        ) << std::format("{:20.12f}\t", physicalData.getIntraEnergy());

        if (settings::GeneralSettings::isMMActivated())
        {
            _getFilePointer()
                << std::format("{:20.12f}\t", physicalData.getCoulombEnergy());
            _getFilePointer(
            ) << std::format("{:20.12f}\t", physicalData.getNonCoulombEnergy());
        }

        if (settings::ForceFieldSettings::isActive())
        {
            _getFilePointer()
                << std::format("{:20.12f}\t", physicalData.getBondEnergy());
            _getFilePointer()
                << std::format("{:20.12f}\t", physicalData.getAngleEnergy());
            _getFilePointer()
                << std::format("{:20.12f}\t", physicalData.getDihedralEnergy());
            _getFilePointer()
                << std::format("{:20.12f}\t", physicalData.getImproperEnergy());
        }

        if (settings::GeneralSettings::isHybridJobtype())
        {
            _getFilePointer() << std::format(
                "{:20.12f}\t",
                physicalData.getNumberOfSmoothingMolecules()
            );
        }

        if (settings.manostatSettings.getManostatType() != ManostatType::NONE)
        {
            _getFilePointer()
                << std::format("{:20.12f}\t", physicalData.getVolume());
            _getFilePointer()
                << std::format("{:20.12f}\t", physicalData.getDensity());

            if (settings.manostatSettings.getFixedAxis() != FixedAxis::NONE)
            {
                _getFilePointer() << std::format(
                    "{:20.12f}\t",
                    physicalData.getCoupledPressure()
                );
            }
        }

        if (settings::ThermostatSettings::getThermostatType() ==
            ThermostatType::NOSE_HOOVER)
        {
            _getFilePointer() << std::format(
                "{:20.12f}\t",
                physicalData.getNoseHooverMomentumEnergy()
            );
            _getFilePointer() << std::format(
                "{:20.12f}\t",
                physicalData.getNoseHooverFrictionEnergy()
            );
        }

        if (settings.constraintSettings.isDistanceConstraintsActivated())
        {
            _getFilePointer() << std::format(
                "{:20.12f}\t",
                physicalData.getLowerDistanceConstraints()
            );
            _getFilePointer() << std::format(
                "{:20.12f}\t",
                physicalData.getUpperDistanceConstraints()
            );
        }

        _getFilePointer(
        ) << std::format("{:20.5e}\t", norm(physicalData.getMomentum()));
        _getFilePointer(
        ) << std::format("{:12.5f}\n", physicalData.getLoopTime());

        _getFilePointer() << std::flush;
    }

}   // namespace out
