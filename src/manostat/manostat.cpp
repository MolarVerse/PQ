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

#include <algorithm>
#include <cmath>

#include "constants/internalConversionFactors.hpp"
#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "globalTimer.hpp"
#include "manostatSettings.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "triclinicBox.hpp"

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

        if (settings::ManostatSettings::getIsotropy() !=
            Isotropy::FULL_ANISOTROPIC)
        {
            // Length coupling uses cell-axis coordinates: T^-1 P T.
            // Full anisotropic deformation acts directly in Cartesian space.
            const auto &box = simulationBox.getBox();
            _pressureTensor = box.toOrthoSpace(_pressureTensor) *
                              box.toSimSpace(linalg::diagonalMatrix(1.0));
        }

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

    /** @brief Validate a resize without changing the live cell or atoms. */
    void Manostat::_validateScaling(
        const molsys::SimulationBox &simulationBox,
        const linalg::tensor3D      &mu
    )
    {
        if (settings::GeneralSettings::getVirialType() == VirialType::ATOMIC &&
            std::ranges::any_of(
                simulationBox.getMolecules(),
                [](const auto &molecule)
                { return molecule.getNumberOfAtoms() > 1; }
            ))
            throw exc::ManostatException(
                "Pressure coupling of multi-atom molecules requires "
                "virial = molecular"
            );

        const auto finiteMatrix = [](const auto &matrix)
        {
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    if (!std::isfinite(matrix[i][j]))
                        return false;
            return true;
        };

        const auto determinant = det(mu);
        if (!finiteMatrix(mu) || !std::isfinite(determinant) ||
            determinant <= 0.0 || mu[0][0] <= 0.0 || mu[1][1] <= 0.0 ||
            mu[2][2] <= 0.0 || !finiteMatrix(inverse(mu)))
            throw exc::ManostatException("Invalid manostat scaling matrix");

        const auto validateCandidate = [&](auto candidate)
        {
            candidate.scaleBox(mu);
            const auto volume     = candidate.calculateVolume();
            const auto dimensions = candidate.getBoxDimensions();
            const auto angles     = candidate.getBoxAngles();
            const auto matrix     = candidate.getBoxMatrix();
            if (!std::isfinite(volume) || volume <= 0.0 ||
                !finiteMatrix(matrix) || !finiteMatrix(inverse(matrix)))
                throw exc::ManostatException("Invalid manostat cell geometry");

            for (size_t i = 0; i < 3; ++i)
                if (!std::isfinite(dimensions[i]) || dimensions[i] <= 0.0 ||
                    !std::isfinite(angles[i]) || angles[i] <= 0.0 ||
                    angles[i] >= 180.0)
                    throw exc::ManostatException(
                        "Invalid manostat cell geometry"
                    );

            if (candidate.getMinimalBoxDimension() <
                2.0 * settings::PotentialSettings::getCoulombRadiusCutOff())
                throw exc::ManostatException(
                    "Coulomb radius cut off is larger than half of the minimal "
                    "box "
                    "dimension"
                );
        };

        const auto &box = simulationBox.getBox();
        if (const auto *triclinic =
                dynamic_cast<const molsys::TriclinicBox *>(&box))
            validateCandidate(*triclinic);
        else
            validateCandidate(
                dynamic_cast<const molsys::OrthorhombicBox &>(box)
            );
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
     * @brief get the manostat type
     *
     * @return ManostatType
     */
    ManostatType Manostat::getManostatType() const
    {
        return ManostatType::NONE;
    }

}   // namespace manostat
