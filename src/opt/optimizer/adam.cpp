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

#include <cmath>   // for pow, sqrt

#include "simulationBox.hpp"

using namespace opt;

/**
 * @brief Constructor
 *
 * @param nIterations
 * @param nAtoms
 */
Adam::Adam(const size_t nEpochs, const size_t nAtoms) : Optimizer(nEpochs)
{
    _momentum1.resize(nAtoms, linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    _momentum2.resize(nAtoms, linearAlgebra::Vec3D(0.0, 0.0, 0.0));
}

/**
 * @brief Constructor
 *
 * @param nIterations
 * @param beta1
 * @param beta2
 */
Adam::Adam(
    const size_t nEpochs,
    const double beta1,
    const double beta2,
    const size_t nAtoms
)
    : Optimizer(nEpochs), _beta1(beta1), _beta2(beta2)
{
    _momentum1.resize(nAtoms, linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    _momentum2.resize(nAtoms, linearAlgebra::Vec3D(0.0, 0.0, 0.0));
}

/**
 * @brief clone the optimizer
 *
 * @return pq::SharedOptimizer
 */
pq::SharedOptimizer Adam::clone() const
{
    return std::make_shared<Adam>(*this);
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
 */
void Adam::update(const double learningRate, const size_t step)
{
    for (size_t i = 0; i < _simulationBox->getNumberOfAtoms(); ++i)
    {
        const auto force = _simulationBox->getAtoms()[i]->getForce();
        const auto pos   = _simulationBox->getAtoms()[i]->getPosition();

        _momentum1[i] = _beta1 * _momentum1[i] - (1.0 - _beta1) * force;
        _momentum2[i] = _beta2 * _momentum2[i] + (1.0 - _beta2) * force * force;

        const auto m1 = _momentum1[i] / (1.0 - std::pow(_beta1, step));
        const auto m2 = _momentum2[i] / (1.0 - std::pow(_beta2, step));

        auto pos_new = pos - learningRate * m1 / (sqrt(m2 + 1e-8));
        _simulationBox->applyPBC(pos_new);

        _simulationBox->getAtoms()[i]->setPositionOld(pos);
        _simulationBox->getAtoms()[i]->setPosition(pos_new);
    }
}