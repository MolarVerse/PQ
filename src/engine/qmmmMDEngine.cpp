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

namespace engine
{
    /**
     * @brief calculate QM/MM forces
     *
     */
    void QMMMMDEngine::calculateForces()
    {
        _configurator.calculateInnerRegionCenter(*_simulationBox);
        _configurator.shiftAtomsToInnerRegionCenter(*_simulationBox);
        _configurator.assignHybridZones(*_simulationBox);
        _configurator.calculateSmoothingFactors(*_simulationBox);

        applyExactSmoothing();
        // TODO: https://github.com/MolarVerse/PQ/issues/198

        _configurator.shiftAtomsBackToInitialPositions(*_simulationBox);
    }

    /**
     * @brief Apply exact smoothing algorithm for QM/MM boundary treatment
     *
     * @details This function implements the exact smoothing algorithm by
     * iterating through all 2^n combinations of smoothing molecules being
     * active/inactive in the inner (QM) region. For each combination:
     * 1. Run QM calculation with selected smoothing molecules
     * 2. Run MM calculation for the complementary set
     * 3. Calculate global smoothing factor for weighted contribution
     *
     * @note Computational cost: O(2^n) where n = number of smoothing molecules
     */
    void QMMMMDEngine::applyExactSmoothing()
    {
        const auto n_SmoothingMolecules =
            _configurator.getNumberSmoothingMolecules();

        // Loop over all combinations of smoothing molecules
        for (size_t i = 0; i < (1u << n_SmoothingMolecules); ++i)
        {
            const auto inactiveForInnerCalcMolecules =
                generateInactiveMoleculeSet(i, n_SmoothingMolecules);

            // STEP 1: Setup and run QM calculation
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

            // STEP 2: Setup and run MM calculation
            _configurator.activateMolecules(*_simulationBox);
            _configurator.deactivateInnerMolecules(*_simulationBox);
            _configurator.activateSmoothingMolecules(
                inactiveForInnerCalcMolecules,
                *_simulationBox
            );

            // TODO: https://github.com/MolarVerse/PQ/issues/195

            // STEP 3: Calculate global smoothing factor for this combination
            const double globalSmoothingFactor =
                calculateGlobalSmoothingFactor(inactiveForInnerCalcMolecules);

            // TODO: https://github.com/MolarVerse/PQ/issues/197
        }
    }

    /**
     * @brief Generate set of inactive smoothing molecule indices from bit
     * pattern
     *
     * @param bitPattern Binary representation where each bit indicates if a
     *                   smoothing molecule should be inactive (1) or active (0)
     * @param totalMolecules Total number of smoothing molecules
     * @return std::unordered_set<size_t> Set of indices to deactivate
     *
     * @details This function converts a bit pattern into a set of molecule
     * indices. Each bit position corresponds to a smoothing molecule index. If
     * bit j is set, molecule j will be included in the inactive set.
     */
    std::unordered_set<size_t> QMMMMDEngine::generateInactiveMoleculeSet(
        size_t bitPattern,
        size_t totalMolecules
    )
    {
        std::unordered_set<size_t> inactiveMolecules;

        for (size_t j = 0; j < totalMolecules; ++j)
            if (bitPattern & (1u << j))
                inactiveMolecules.insert(j);

        return inactiveMolecules;
    }

    /**
     * @brief Calculate global smoothing factor for QM/MM boundary treatment
     *
     * @param inactiveForInnerCalcMolecules Set of molecule indices that are
     *                                      inactive for inner calculation
     * @return double Global smoothing factor for weighted contribution
     *
     * @details This function calculates the global smoothing factor by
     * iterating through all smoothing molecules and multiplying their
     * individual smoothing factors. For molecules marked as inactive for
     * inner calculation, it uses (1 - smoothingFactor), otherwise it uses
     * the smoothingFactor directly.
     */
    double QMMMMDEngine::calculateGlobalSmoothingFactor(
        const std::unordered_set<size_t>& inactiveForInnerCalcMolecules
    )
    {
        using enum simulationBox::HybridZone;

        double globalSmoothingFactor = 1.0;

        size_t index = 0;
        for (const auto& mol :
             _simulationBox->getMoleculesInsideZone(SMOOTHING))
        {
            if (inactiveForInnerCalcMolecules.contains(index))
                globalSmoothingFactor *= 1 - mol.getSmoothingFactor();
            else
                globalSmoothingFactor *= mol.getSmoothingFactor();

            ++index;
        }

        return globalSmoothingFactor;
    }

}   // namespace engine