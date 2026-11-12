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

#include <cmath>

#include "box.hpp"
#include "constants/internalConversionFactors.hpp"
#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "globalTimer.hpp"
#include "manostatSettings.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "unit_types.hpp"
#include "units.hpp"

namespace manostat
{

    /**
     * @brief Construct a new Manostat:: Manostat object
     *
     * @param settings The settings object containing the manostat configuration
     */
    Manostat::Manostat(const ManostatSettings &settings)
        : _settings(settings), _pressureTensor{0.0 * Pa}, _pressure{0.0 * Pa}
    {
    }

    /**
     * @brief calculate the pressure of the system
     *
     * @param simulationBox The simulation box containing the system
     * @param physicalData The physical data of the system
     */
    void Manostat::calculatePressure(
        const molsys::SimulationBox &simulationBox,
        physicalData::PhysicalData  &physicalData
    )
    {
        const auto ekinVirial = physicalData.getKinEnergyVirialTensor(
            settings::GeneralSettings::getVirialType()
        );
        const auto forceVirial = physicalData.getVirial();
        const auto volume      = simulationBox.getVolume();

        auto pressureTensor  = (2.0 * ekinVirial + forceVirial) / volume;
        pressureTensor      *= PRESSURE_FACTOR_PA;
        _pressureTensor      = pressureTensor * Pa;
        _pressure            = trace(_pressureTensor) / linalg::tensor3D::size;

        physicalData.setPressure(_pressure);

        const auto &box = simulationBox.getBox();
        if (!box.allowChangeOfAngles() && !box.isOrthorhombic())
        {
            // Length coupling uses cell-axis coordinates: T^-1 P T.
            // Full anisotropic deformation acts directly in Cartesian space.
            _pressureTensor = box.getInverseBoxMatrix() * _pressureTensor *
                              box.getBoxMatrix();
        }

        const auto fixedAxis = _settings->getFixedAxis();
        const auto p_xyz     = diagonal(_pressureTensor);

        size_t   numFree = 0;
        Pressure p_avg{0.0 * Pa};

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
     * @brief transform a scaling tensor into simulation space
     *
     * @details for boxes with fixed angles (non-orthorhombic) the scaling
     * tensor is diagonal in orthogonal space and is transformed via
     * h S h^-1 (equal to T S T^-1 for diagonal S). For orthorhombic boxes and
     * boxes allowing a change of angles the tensor already acts in simulation
     * space.
     *
     * @param scalingTensor
     * @param box
     * @return QuantityTensor3D representing the scaling tensor transformed into
     * simulation space
     */
    QuantityTensor3D Manostat::_toSimSpaceScaling(
        const QuantityTensor3D &scalingTensor,
        const molsys::Box      &box
    )
    {
        if (box.allowChangeOfAngles() || box.isOrthorhombic())
            return scalingTensor;

        return box.getBoxMatrix() * scalingTensor * box.getInverseBoxMatrix();
    }

    /**
     * @brief Validate a resize without changing the live cell or atoms.
     *
     * @param simulationBox The current simulation box.
     * @param mu The scaling tensor to validate.
     *
     * @throws exc::ManostatException if the scaling tensor is invalid or leads
     * to an invalid candidate cell.
     */
    void Manostat::_validateScaling(
        const molsys::SimulationBox &simulationBox,
        const QuantityTensor3D      &mu
    )
    {
        const auto determinant = det(mu);
        if (!linalg::isFinite(mu) ||
            !std::isfinite(static_cast<double>(determinant)) ||
            determinant <= 0.0 || mu[0][0] <= 0.0 || mu[1][1] <= 0.0 ||
            mu[2][2] <= 0.0 || !linalg::isFinite(inverse(mu)))
        {
            throw exc::ManostatException("Invalid manostat scaling matrix");
        }

        auto candidate = simulationBox.getBox().clone();
        _validateCandidate(*candidate, mu);
    }

    /**
     * @brief Scale and validate a candidate cell.
     *
     * @param candidate The candidate simulation box to scale and validate.
     * @param mu The scaling tensor to apply to the candidate box.
     *
     * @throws exc::ManostatException if the resulting candidate cell geometry
     * is invalid.
     */
    void Manostat::_validateCandidate(
        molsys::Box            &candidate,
        const QuantityTensor3D &mu
    )
    {
        candidate.scaleBox(mu);
        const auto volume     = candidate.getVolume();
        const auto dimensions = candidate.get_boxDimensions();
        const auto angles     = candidate.getBoxAngles();
        const auto matrix = candidate.getBoxMatrix(molsys::Periodicity::XYZ);
        if (!std::isfinite(volume) || volume <= 0.0 ||
            !linalg::isFinite(matrix) || !linalg::isFinite(inverse(matrix)))
        {
            throw exc::ManostatException("Invalid manostat cell geometry");
        }

        constexpr double straightAngle = 180.0;
        for (size_t i = 0; i < 3; ++i)
        {
            if (!std::isfinite(dimensions[i]) || dimensions[i] <= 0.0 ||
                !std::isfinite(angles[i]) || angles[i] <= 0.0 ||
                angles[i] >= straightAngle)
            {
                throw exc::ManostatException("Invalid manostat cell geometry");
            }
        }

        if (candidate.getMinimalBoxDimension() <
            2.0 * settings::PotentialSettings::getCoulombRadiusCutOff())
        {
            throw exc::ManostatException(
                "Coulomb radius cut off is larger than half of the minimal "
                "box dimension"
            );
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
    void Manostat::rotateMu(QuantityTensor3D &mu)
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
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const Temperature & /*targetTemperature*/
    )
    {
        auto _ = scopedTimer(TimerId::Manostat, "Calc Pressure");

        calculatePressure(simulationBox, physicalData);
    }

    /**
     * @brief get the pressure tensor in pascal
     *
     * @return const PressureTensor3D&
     */
    const PressureTensor3D &Manostat::_getPressureTensor() const
    {
        return _pressureTensor;
    }

    /**
     * @brief get the pressure in pascal
     *
     * @return const Pressure&
     */
    const Pressure &Manostat::_getPressure() const { return _pressure; }

    /**
     * @brief set the pressure tensor
     *
     * @param pTensor The new pressure tensor
     */
    void Manostat::_setPressureTensor(const PressureTensor3D &pTensor)
    {
        _pressureTensor = pTensor;
    }

    /**
     * @brief set the pressure
     *
     * @param pressure The new pressure
     */
    void Manostat::_setPressure(const Pressure &pressure)
    {
        _pressure = pressure;
    }

    /**
     * @brief get the manostat settings
     *
     * @return const ManostatSettings&
     */
    const ManostatSettings &Manostat::_getSettings() const
    {
        return _settings.get();
    }

}   // namespace manostat
