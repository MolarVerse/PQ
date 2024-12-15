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

#include <functional>   // for function

#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox

using namespace manostat;
using namespace simulationBox;
using namespace physicalData;
using namespace constants;
using namespace settings;
using namespace linearAlgebra;

/**
 * @brief Construct a new Manostat:: Manostat object
 *
 * @param targetPressure
 */
Manostat::Manostat(const double targetPressure)
    : _targetPressure(targetPressure)
{
}

/**
 * @brief calculate the pressure of the system
 *
 * @param data
 */
void Manostat::calculatePressure(const SimulationBox &box, PhysicalData &data)
{
    auto       ekinVirial  = data.getKinEnergyVirialTensor();
    auto       forceVirial = data.getVirial();
    const auto volume      = box.getVolume();

    ekinVirial  = box.getBox().toOrthoSpace(ekinVirial);
    forceVirial = box.getBox().toOrthoSpace(forceVirial);

    _pressureTensor  = (2.0 * ekinVirial + forceVirial) / volume;
    _pressureTensor *= _PRESSURE_FACTOR_;

    _pressure = trace(_pressureTensor) / 3.0;

    data.setPressure(_pressure);
}

/**
 * @brief rotate mu back into upper diagonal space
 *
 * @details first order approximation of mu rotation according to gromacs
 * @link https://manual.gromacs.org/current/reference-manual/algorithms/molecular-dynamics.html
 *
 * @param mu
 */
void Manostat::rotateMu(tensor3D &mu)
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
 * @param data
 */
void Manostat::applyManostat(SimulationBox &box, PhysicalData &data)
{
    startTimingsSection("Calc Pressure");

    calculatePressure(box, data);

    stopTimingsSection("Calc Pressure");
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