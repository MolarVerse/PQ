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

#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "globalTimer.hpp"
#include "manostatSettings.hpp"   // for ManostatType, Isotropy
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox

using namespace manostat;
using namespace molsys;
using namespace physicalData;
using namespace constants;
using namespace settings;
using namespace linearAlgebra;

/**
 * @brief Construct a new Manostat:: Manostat object
 *
 * @param targetPressure
 */
Manostat::Manostat(double targetPressure) : _targetPressure(targetPressure) {}

/**
 * @brief calculate the pressure of the system
 *
 * @param box
 * @param data
 */
void Manostat::calculatePressure(const SimulationBox& box, PhysicalData& data)
{
    auto ekinVirial =
        data.getKinEnergyVirialTensor(settings::Settings::getVirialType());
    auto       forceVirial = data.getVirial();
    const auto volume      = box.getVolume();

    ekinVirial  = box.getBox().toOrthoSpace(ekinVirial);
    forceVirial = box.getBox().toOrthoSpace(forceVirial);

    _pressureTensor  = (2.0 * ekinVirial + forceVirial) / volume;
    _pressureTensor *= PRESSURE_FACTOR;
    _pressure        = trace(_pressureTensor) / linearAlgebra::tensor3D::size;

    data.setPressure(_pressure);

    const auto fixedAxis = ManostatSettings::getFixedAxis();
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
        data.setCoupledPressure(p_avg);
    }
    else
    {
        data.setCoupledPressure(_pressure);
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
void Manostat::rotateMu(tensor3D& mu)
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
 * @param box
 * @param data
 */
void Manostat::applyManostat(SimulationBox& box, PhysicalData& data)
{
    auto _ = scopedTimer(TimerId::Manostat, "Calc Pressure");

    calculatePressure(box, data);
}

/**
 * @brief get the manostat type
 *
 * @return ManostatType
 */
ManostatType Manostat::getManostatType() const { return ManostatType::NONE; }

/**
 * @brief get the isotropy of the manostat
 *
 * @return Isotropy
 */
Isotropy Manostat::getIsotropy() const { return Isotropy::NONE; }
