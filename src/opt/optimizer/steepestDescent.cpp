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

#include "steepestDescent.hpp"

#include "optimizer.hpp"
#include "simulationBox.hpp"

using namespace opt;

/**
 * @brief Constructor
 *
 * @param nIterations
 * @param learningRate
 */
SteepestDescent::SteepestDescent(const size_t nEpochs) : Optimizer(nEpochs) {}

/**
 * @brief clone the optimizer
 *
 * @return std::shared_ptr<Optimizer>
 */
std::shared_ptr<Optimizer> SteepestDescent::clone() const
{
    return std::make_shared<SteepestDescent>(*this);
}

/**
 * @brief get the maximum history length
 *
 * @return size_t
 */
size_t SteepestDescent::maxHistoryLength() const { return _maxHistoryLength; }

/**
 * @brief update the optimizer
 *
 * @param learningRate
 */
void SteepestDescent::update(const double learningRate)
{
    const auto &atoms = _simulationBox->getAtoms();

    for (auto &atom : atoms)
    {
        const auto force = atom->getForce();

        atom->setPositionOld(atom->getPosition());
        atom->addPosition(learningRate * force);
    }

    updateHistory();
}