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

#include <format>   // for format
#include <ranges>   // for distance

#include "exceptions.hpp"         // for HybridMDEngineException
#include "hybridSettings.hpp"     // for HybridSettings
#include "manostatSettings.hpp"   // for ManostatType

using namespace settings;
using namespace customException;

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

        // Set number of QM atoms in physical data for output purposes
        setNumberOfQMAtoms();

        const auto& smoothingMethod = HybridSettings::getSmoothingMethod();
        using enum SmoothingMethod;

        // TODO: https://github.com/MolarVerse/PQ/issues/198
        if (smoothingMethod == HOTSPOT)
            throw HybridMDEngineException(
                "Hotspot smoothing method not yet implemented"
            );
        else if (smoothingMethod == EXACT)
            applyExactSmoothing();
        else
            throw HybridMDEngineException("Unknown smoothing method requested");

        _configurator.shiftAtomsBackToInitialPositions(*_simulationBox);
    }

    /**
     * @brief Apply exact smoothing algorithm for QM/MM boundary treatment
     *
     * @details This function implements the exact smoothing algorithm by
     * iterating through all 2^n combinations of smoothing molecules being
     * active/inactive in the inner (QM) region.
     *
     * @note Computational cost: O(2^n) where n = number of smoothing molecules
     */
    void QMMMMDEngine::applyExactSmoothing()
    {
        using enum simulationBox::HybridZone;
        using std::ranges::distance;

        auto         qmEnergy         = 0.0;
        auto         coulombEnergy    = 0.0;
        auto         nonCoulombEnergy = 0.0;
        pq::tensor3D virial           = {0.0};

        auto& atoms = _simulationBox->getAtoms();

        const auto nSmMol =
            distance(_simulationBox->getMoleculesInsideZone(SMOOTHING));

        // Loop over all combinations of smoothing molecules
        for (size_t i = 0; i < (1u << nSmMol); ++i)
        {
            // STEP 1: Generate set of inactive molecules and calculate
            // associated global smoothing factor for this configuration
            const auto inactiveSmMol = generateInactiveMoleculeSet(i, nSmMol);

            const auto globalSmF =
                calculateGlobalSmoothingFactor(inactiveSmMol);

            // STEP 2: Setup and run QM calculation, accumulate QM forces and QM
            // virial contribution
            _configurator.activateMolecules(*_simulationBox);
            _configurator.deactivateOuterMolecules(*_simulationBox);
            _configurator.deactivateSmoothingMolecules(
                inactiveSmMol,
                *_simulationBox
            );

            _qmRunner->run(
                *_simulationBox,
                *_physicalData,
                simulationBox::Periodicity::NON_PERIODIC
            );

            if (ManostatSettings::getManostatType() != ManostatType::NONE)
                virial +=
                    _virial->calculateQMVirial(*_simulationBox) * globalSmF;

            for (auto& atom : atoms)
            {
                const auto force = atom->getForce();
                atom->addForceInner(force * globalSmF);
            }
            _simulationBox->resetForces();

            // STEP 3: Setup and run MM calculation, accumulate MM forces and MM
            // virial contribution
            _configurator.toggleMoleculeActivation(*_simulationBox);

            _potential->calculateQMMMForces(
                *_simulationBox,
                *_physicalData,
                *_cellList
            );

            // TODO: https://github.com/MolarVerse/PQ/issues/195

            for (auto& atom : atoms)
            {
                const auto force = atom->getForce();
                atom->addForceOuter(force * globalSmF);
            }
            _simulationBox->resetForces();

            if (ManostatSettings::getManostatType() != ManostatType::NONE)
            {
                _virial->calculateVirial(*_simulationBox, *_physicalData);
                virial += _physicalData->getVirial() * globalSmF;
            }

            // Scale and accumulate energies
            qmEnergy      += _physicalData->getQMEnergy() * globalSmF;
            coulombEnergy += _physicalData->getCoulombEnergy() * globalSmF;
            nonCoulombEnergy +=
                _physicalData->getNonCoulombEnergy() * globalSmF;
        }

        // STEP 4: Set forces to the accumulated hybrid forces for MD routine
        for (auto& atom : atoms)
        {
            const auto innerForce = atom->getForceInner();
            const auto outerForce = atom->getForceOuter();
            atom->setForce(innerForce + outerForce);
        }

        // STEP 5: Set energies to the accumulated hybrid energies
        _physicalData->setQMEnergy(qmEnergy);
        _physicalData->setCoulombEnergy(coulombEnergy);
        _physicalData->setNonCoulombEnergy(nonCoulombEnergy);
        _physicalData->setVirial(virial);
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

    /**
     * @brief Set the number of QM atoms in physical data for output purposes
     *
     * @details This function temporarily configures the simulation box to count
     * only QM atoms by activating all molecules and then deactivating outer
     * molecules. The count is stored in physical data and then all molecules
     * are reactivated to restore the original state.
     */
    void QMMMMDEngine::setNumberOfQMAtoms()
    {
        _configurator.activateMolecules(*_simulationBox);
        _configurator.deactivateOuterMolecules(*_simulationBox);
        _physicalData->setNumberOfQMAtoms(_simulationBox->getNumberOfQMAtoms());
        _configurator.activateMolecules(*_simulationBox);
    }

}   // namespace engine