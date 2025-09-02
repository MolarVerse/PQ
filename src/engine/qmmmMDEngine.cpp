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

#include "qmmmMDEngine.hpp"

#include <iterator>   // for std::ranges:distance

namespace engine
{
    /**
     * @brief calculate QM/MM forces
     *
     */
    void QMMMMDEngine::calculateForces()
    {
        using std::ranges::distance;
        using enum simulationBox::HybridZone;

        _configurator.calculateInnerRegionCenter(*_simulationBox);
        _configurator.shiftAtomsToInnerRegionCenter(*_simulationBox);
        _configurator.assignHybridZones(*_simulationBox);
        _configurator.calculateSmoothingFactors(*_simulationBox);

        const auto n_SmoothingMolecules =
            _configurator.getNumberSmoothingMolecules();

        // exact smoothing: loop over all combinations of smoothing molecules
        for (size_t i = 0; i < (1u << n_SmoothingMolecules); ++i)
        {
            std::unordered_set<size_t> inactiveForInnerCalcMolecules;

            for (size_t j = 1; j < n_SmoothingMolecules; ++j)
                if (i & (1u << j))
                    inactiveForInnerCalcMolecules.insert(j);

            _configurator.activateMolecules(*_simulationBox);
            _configurator.deactivateOuterMolecules(*_simulationBox);
            _configurator.deactivateSmoothingMolecules(
                inactiveForInnerCalcMolecules,
                *_simulationBox
            );

            _qmRunner->run(
                *_simulationBox,
                *_physicalData,
                simulationBox::Periodicity::NON_PERIODIC
            );

            _configurator.activateMolecules(*_simulationBox);
            _configurator.deactivateInnerMolecules(*_simulationBox);
            _configurator.activateSmoothingMolecules(
                inactiveForInnerCalcMolecules,
                *_simulationBox
            );

            // TODO: https://github.com/MolarVerse/PQ/issues/195

            double globalSmoothingFactor = 1.0;

            size_t index{0};
            for (const auto& mol :
                 _simulationBox->getMoleculesInsideZone(SMOOTHING))
            {
                if (inactiveForInnerCalcMolecules.contains(index))
                    globalSmoothingFactor *= 1 - mol.getSmoothingFactor();
                else
                    globalSmoothingFactor *= mol.getSmoothingFactor();

                ++index;
            }

            // TODO: https://github.com/MolarVerse/PQ/issues/197
        }

        // TODO: https://github.com/MolarVerse/PQ/issues/198

        _configurator.shiftAtomsBackToInitialPositions(*_simulationBox);
    }

}   // namespace engine