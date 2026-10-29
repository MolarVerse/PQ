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

namespace manostat
{

    /**
     * @brief Construct a new Manostat:: Manostat object
     *
     * @param settings The settings object containing the manostat configuration
     */
    Manostat::Manostat(const ManostatSettings &settings) : _settings(settings)
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

        _pressureTensor  = (2.0 * ekinVirial + forceVirial) / volume;
        _pressureTensor *= PRESSURE_FACTOR;
        _pressure        = trace(_pressureTensor) / linalg::tensor3D::size;

        physicalData.setPressure(_pressure);

        if (_settings->getIsotropy() != Isotropy::FULL_ANISOTROPIC)
        {
            // Length coupling uses cell-axis coordinates: T^-1 P T.
            // Full anisotropic deformation acts directly in Cartesian space.
            const auto &box = simulationBox.getBox();
            _pressureTensor = box.toOrthoSpace(_pressureTensor) *
                              box.toSimSpace(linalg::diagonalMatrix(1.0));
        }

        const auto fixedAxis = _settings->getFixedAxis();
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

    /** @brief Validate a resize without changing the live cell or atoms. */
    void Manostat::_validateScaling(
        const molsys::SimulationBox &simulationBox,
        const linalg::tensor3D      &mu
    )
    {
        const auto determinant = det(mu);
        if (!linalg::isFinite(mu) || !std::isfinite(determinant) ||
            determinant <= 0.0 || mu[0][0] <= 0.0 || mu[1][1] <= 0.0 ||
            mu[2][2] <= 0.0 || !linalg::isFinite(inverse(mu)))
        {
            throw exc::ManostatException("Invalid manostat scaling matrix");
        }

        auto candidate = simulationBox.getBox().clone();
        _validateCandidate(*candidate, mu);
    }

    /** @brief Scale and validate a candidate cell. */
    void Manostat::_validateCandidate(
        molsys::Box            &candidate,
        const linalg::tensor3D &mu
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
    void Manostat::rotateMu(linalg::tensor3D &mu)
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
        molsys::SimulationBox &simulationBox,

        physicalData::PhysicalData &physicalData

    )
    {
        auto _ = scopedTimer(TimerId::Manostat, "Calc Pressure");

        calculatePressure(simulationBox, physicalData);
    }

    /**
     * @brief get the pressure tensor
     *
     * @return const linalg::tensor3D&
     */
    const linalg::tensor3D &Manostat::_getPressureTensor() const
    {
        return _pressureTensor;
    }

    /**
     * @brief get the pressure
     *
     * @return double
     */
    double Manostat::_getPressure() const { return _pressure; }

    /**
     * @brief set the pressure tensor
     *
     * @param pTensor The new pressure tensor
     */
    void Manostat::_setPressureTensor(const linalg::tensor3D &pTensor)
    {
        _pressureTensor = pTensor;
    }

    /**
     * @brief set the pressure
     *
     * @param pressure The new pressure
     */
    void Manostat::_setPressure(double pressure) { _pressure = pressure; }

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
