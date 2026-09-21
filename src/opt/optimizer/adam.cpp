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

#include "adam.hpp"

#include <cmath>

#include "simulationBox.hpp"

using namespace opt;

/**
 * @brief Constructor
 *
 * @param nEpochs
 * @param nAtoms
 */
Adam::Adam(size_t nEpochs, size_t nAtoms) : Optimizer(nEpochs)
{
    _momentum1.resize(nAtoms, linalg::Vec3D(0.0, 0.0, 0.0));
    _momentum2.resize(nAtoms, linalg::Vec3D(0.0, 0.0, 0.0));
}

/**
 * @brief Constructor
 *
 * @param nEpochs
 * @param beta1
 * @param beta2
 * @param nAtoms
 */
Adam::Adam(size_t nEpochs, double beta1, double beta2, size_t nAtoms)
    : Optimizer(nEpochs), _beta1(beta1), _beta2(beta2)
{
    _momentum1.resize(nAtoms, linalg::Vec3D(0.0, 0.0, 0.0));
    _momentum2.resize(nAtoms, linalg::Vec3D(0.0, 0.0, 0.0));
}

/**
 * @brief get the maximum history length
 *
 * @return size_t
 */
size_t Adam::maxHistoryLength() const { return _maxHistoryLength; }

/**
 * @brief update the optimizer
 *
 * @param learningRate
 * @param step
 */
void Adam::update(double learningRate, size_t step)
{
    auto& simulationBox = _getSimulationBox();

    for (size_t i = 0; i < simulationBox.getNumberOfAtoms(); ++i)
    {
        auto&      atom  = simulationBox.getAtoms()[i];
        const auto force = atom->getForce();
        const auto pos   = atom->getPosition();

        _momentum1[i] = _beta1 * _momentum1[i] - (1.0 - _beta1) * force;
        _momentum2[i] = _beta2 * _momentum2[i] + (1.0 - _beta2) * force * force;

        const auto mom1 = _momentum1[i] / (1.0 - std::pow(_beta1, step));
        const auto mom2 = _momentum2[i] / (1.0 - std::pow(_beta2, step));

        constexpr auto epsilon = 1e-8;
        auto pos_new = pos - learningRate * mom1 / (sqrt(mom2 + epsilon));

        simulationBox.applyPBC(pos_new);

        atom->setPositionOld(pos);
        atom->setPosition(pos_new);
    }
}
