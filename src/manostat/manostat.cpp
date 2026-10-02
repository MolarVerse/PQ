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

#include "manostat.hpp"

#include "constants/internalConversionFactors.hpp"
#include "enums/manostat.hpp"
#include "globalTimer.hpp"
#include "manostatSettings.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

namespace manostat
{

    /**
     * @brief Construct a new Manostat:: Manostat object
     *
     * @param targetPressure
     */
    Manostat::Manostat(double targetPressure) : _targetPressure(targetPressure)
    {
    }

    /**
     * @brief calculate the pressure of the system
     *
     * @param simulationBox The simulation box containing the system
     * @param physicalData The physical data of the system
     */
    void Manostat::calculatePressure(
        const molsys::SimulationBox& simulationBox,
        physicalData::PhysicalData&  physicalData
    )
    {
        auto ekinVirial = physicalData.getKinEnergyVirialTensor(
            settings::Settings::getVirialType()
        );
        auto       forceVirial = physicalData.getVirial();
        const auto volume      = simulationBox.getVolume();

        ekinVirial  = simulationBox.getBox().toOrthoSpace(ekinVirial);
        forceVirial = simulationBox.getBox().toOrthoSpace(forceVirial);

        _pressureTensor  = (2.0 * ekinVirial + forceVirial) / volume;
        _pressureTensor *= PRESSURE_FACTOR;
        _pressure        = trace(_pressureTensor) / linalg::tensor3D::size;

        physicalData.setPressure(_pressure);

        const auto fixedAxis = settings::ManostatSettings::getFixedAxis();
        const auto p_xyz     = diagonal(_pressureTensor);

        size_t numFree = 0;
        double p_avg   = 0.0;

        for (size_t axis = 0; axis < 3; ++axis)
        {
            if (!isAxisFixed(fixedAxis, axis))
            {
                p_avg += p_xyz[axis];
                ++numFree;
            }
        }

        if (numFree > 0)
        {
            p_avg /= static_cast<double>(numFree);
            physicalData.setCoupledPressure(p_avg);
        }
        else
        {
            physicalData.setCoupledPressure(_pressure);
        }
    }

    /**
     * @brief rotate mu back into upper diagonal space
     *
     * @param mu
     *
     * @details first order approximation of mu rotation according to
     * [gromacs](https://manual.gromacs.org/current/reference-manual/algorithms/molecular-dynamics.html)
     *
     */
    void Manostat::rotateMu(linalg::tensor3D& mu)
    {
        mu[0][1] += mu[1][0];
        mu[0][2] += mu[2][0];
        mu[1][2] += mu[2][1];

        mu[1][0] = 0.0;
        mu[2][0] = 0.0;
        mu[2][1] = 0.0;
    }

    /**
     * @brief apply dummy manostat for NVT ensemble
     *
     * @param simulationBox The simulation box containing the system
     * @param physicalData The physical data of the system
     */
    void Manostat::applyManostat(
        molsys::SimulationBox& simulationBox,

        physicalData::PhysicalData& physicalData

    )
    {
        auto _ = scopedTimer(TimerId::Manostat, "Calc Pressure");

        calculatePressure(simulationBox, physicalData);
    }

    /**
     * @brief get the manostat type
     *
     * @return ManostatType
     */
    ManostatType Manostat::getManostatType() const
    {
        return ManostatType::NONE;
    }

}   // namespace manostat
