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

using namespace pq;
using namespace customException;
using namespace settings;
using namespace simulationBox;

using enum SmoothingMethod;
using enum HybridZone;

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
        moltypeCheck();
        _configurator.calculateSmoothingFactors(*_simulationBox);

        // Set number of QM atoms in physical data for output purposes
        setNumberOfQMAtoms();

        const auto& smoothingMethod = HybridSettings::getSmoothingMethod();

        // TODO: https://github.com/MolarVerse/PQ/issues/202
        if (smoothingMethod == HOTSPOT)
            applyHotspotSmoothing();
        else if (smoothingMethod == EXACT)
            applyExactSmoothing();
        else
            throw HybridMDEngineException("Unknown smoothing method requested");

        for (auto& atom : _simulationBox->getAtoms())
        {
            const auto innerForce = atom->getForceInner();
            const auto outerForce = atom->getForceOuter();
            atom->setForce(innerForce + outerForce);
        }

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
        using enum Periodicity;
        using std::ranges::distance;

        auto       qmEnergy         = 0.0;
        auto       coulombEnergy    = 0.0;
        auto       nonCoulombEnergy = 0.0;
        tensor3D   virial           = {0.0};
        auto&      atoms            = _simulationBox->getAtoms();
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

            _qmRunner->run(*_simulationBox, *_physicalData, NON_PERIODIC);

            virial += _virial->calculateQMVirial(*_simulationBox) * globalSmF;
            accumulateInnerForces(atoms, globalSmF);

            // STEP 3: Setup and run MM calculation, accumulate MM forces and MM
            // virial contribution
            _configurator.toggleMoleculeActivation(*_simulationBox);

            _potential->calculateQMMMForces(
                *_simulationBox,
                *_physicalData,
                *_cellList
            );

            _intraNonBonded->calculate(*_simulationBox, *_physicalData);

            _virial->calculateVirial(*_simulationBox, *_physicalData);
            virial += _physicalData->getVirial() * globalSmF;

            // bonded interactions directly add to physical data virial
            _physicalData->setVirial({0.0});

            _forceField->calculateBondedInteractions(
                *_simulationBox,
                *_physicalData
            );

            virial += _physicalData->getVirial() * globalSmF;
            accumulateOuterForces(atoms, globalSmF);

            // STEP 4: Scale and accumulate hybrid energies
            qmEnergy      += _physicalData->getQMEnergy() * globalSmF;
            coulombEnergy += _physicalData->getCoulombEnergy() * globalSmF;
            nonCoulombEnergy +=
                _physicalData->getNonCoulombEnergy() * globalSmF;
        }

        // STEP 5: Set energies and virial to their accumulated values
        _physicalData->setQMEnergy(qmEnergy);
        _physicalData->setCoulombEnergy(coulombEnergy);
        _physicalData->setNonCoulombEnergy(nonCoulombEnergy);
        _physicalData->setVirial(virial);
    }

    /**
     * @brief Apply hotspot smoothing algorithm for QM/MM boundary treatment
     *
     * @details This function implements the hotspot smoothing algorithm by
     * running separate QM and MM calculations and scaling forces of smoothing
     * molecules according to their individual smoothing factors. More
     * computationally efficient than exact smoothing but less rigorous.
     *
     * @warning The energies yielded by this smoothing method are not correct
     *
     * @note Computational cost: O(1) - constant time regardless of number of
     * smoothing molecules
     */
    void QMMMMDEngine::applyHotspotSmoothing()
    {
        using enum Periodicity;

        auto&    atoms  = _simulationBox->getAtoms();
        tensor3D virial = {0.0};

        // STEP 1: Setup and run QM calculation, scale forces of smoothing
        // molecules with smF
        _configurator.activateMolecules(*_simulationBox);
        _configurator.deactivateOuterMolecules(*_simulationBox);

        _qmRunner->run(*_simulationBox, *_physicalData, NON_PERIODIC);

        scaleSmoothingMoleculeForcesInner();
        virial += _virial->calculateQMVirial(*_simulationBox);
        accumulateInnerForces(atoms);

        // STEP 2: Setup and run inter-nonbonded calculation between
        // MM-MM , CORE-MM , LAYER+SMOOTHING-MM and scale forces of smoothing
        // molecules with smF
        _configurator.toggleMoleculeActivation(*_simulationBox);

        _potential
            ->calculateQMMMForces(*_simulationBox, *_physicalData, *_cellList);

        scaleSmoothingMoleculeForcesInner();
        _virial->calculateVirial(*_simulationBox, *_physicalData);
        virial += _physicalData->getVirial();
        accumulateOuterForces(atoms);

        // STEP 3: Calculate inter-nonbonded forces between SMOOTHING molecules
        // and scale forces of smoothing molecules with (1 - smF)

        _potential->calculateHotspotSmoothingMMForces(
            *_simulationBox,
            *_physicalData,
            *_cellList
        );

        scaleSmoothingMoleculeForcesOuter();
        _virial->calculateVirial(*_simulationBox, *_physicalData);
        virial += _physicalData->getVirial();
        accumulateOuterForces(atoms);

        // STEP 4: Setup and run intra-nonbonded calculation and scale forces of
        // smoothing molecules with (1 - smF)

        _configurator.activateSmoothingMolecules(*_simulationBox);

        _intraNonBonded->calculate(*_simulationBox, *_physicalData);

        scaleSmoothingMoleculeForcesOuter();
        _virial->calculateVirial(*_simulationBox, *_physicalData);
        virial += _physicalData->getVirial();
        accumulateOuterForces(atoms);

        // STEP 5: Run intra-bonded calculation and scale forces of
        // smoothing molecules with (1 - smF)

        // bonded interactions directly add to physical data virial
        _physicalData->setVirial({0.0});

        _forceField->calculateBondedInteractions(
            *_simulationBox,
            *_physicalData
        );

        virial += _physicalData->getVirial();
        scaleSmoothingMoleculeForcesOuter();
        accumulateOuterForces(atoms);

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
        using enum HybridZone;

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

    /**
     * @brief Validates that all non-core molecules have a non-zero moltype
     *
     * @details This function iterates through all molecules and verifies that
     * each molecule outside the QM core has a moltype that is not zero.
     *
     * @throws HybridMDEngineException if any non-core molecule has moltype == 0
     */
    void QMMMMDEngine::moltypeCheck()
    {
        size_t count = 0;
        for (const auto& mol : _simulationBox->getMolecules())
        {
            if (mol.getMoltype() == 0 && mol.getHybridZone() != CORE)
                throw(HybridMDEngineException(
                    std::format(
                        "Molecule number {} is outside the QM core and has "
                        "moltype 0. All molecules outside the QM core must "
                        "have a non-zero moltype assigned.",
                        count
                    )
                ));
            ++count;
        }
    }

    /**
     * @brief Accumulate current forces to inner force storage and reset forces
     *
     * @param atoms Vector of atoms whose forces should be accumulated
     *
     * @details This function copies the current force on each atom to the inner
     * force accumulator using addForceInner(), then resets all forces in the
     * simulation box to zero.
     */
    void QMMMMDEngine::accumulateInnerForces(pq::SharedAtomVec& atoms)
    {
        for (auto& atom : atoms)
        {
            const auto force = atom->getForce();
            atom->addForceInner(force);
        }

        _simulationBox->resetForces();
    }

    /**
     * @brief Accumulate scaled forces to inner force storage and reset forces
     *
     * @param atoms Vector of atoms whose forces should be accumulated
     * @param globalSmF Global smoothing factor to scale forces before
     * accumulation
     *
     * @details This function copies the current force on each atom, scales it
     * by the global smoothing factor, then adds it to the inner force
     * accumulator using addForceInner(). After accumulation, all forces in the
     * simulation box are reset to zero. This overload is used in exact
     * smoothing where forces need to be weighted by the configuration-specific
     * global smoothing factor.
     */
    void QMMMMDEngine::accumulateInnerForces(
        pq::SharedAtomVec& atoms,
        const double       globalSmF
    )
    {
        for (auto& atom : atoms)
        {
            const auto force = atom->getForce();
            atom->addForceInner(force * globalSmF);
        }

        _simulationBox->resetForces();
    }

    /**
     * @brief Accumulate current forces to outer force storage and reset forces
     *
     * @param atoms Vector of atoms whose forces should be accumulated
     *
     * @details This function copies the current force on each atom to the outer
     * force accumulator using addForceOuter(), then resets all forces in the
     * simulation box to zero.
     */
    void QMMMMDEngine::accumulateOuterForces(pq::SharedAtomVec& atoms)
    {
        for (auto& atom : atoms)
        {
            const auto force = atom->getForce();
            atom->addForceOuter(force);
        }

        _simulationBox->resetForces();
    }

    /**
     * @brief Accumulate scaled forces to outer force storage and reset forces
     *
     * @param atoms Vector of atoms whose forces should be accumulated
     * @param globalSmF Global smoothing factor to scale forces before
     * accumulation
     *
     * @details This function copies the current force on each atom, scales it
     * by the global smoothing factor, then adds it to the outer force
     * accumulator using addForceOuter(). After accumulation, all forces in the
     * simulation box are reset to zero. This overload is used in exact
     * smoothing where forces need to be weighted by the configuration-specific
     * global smoothing factor.
     */
    void QMMMMDEngine::accumulateOuterForces(
        pq::SharedAtomVec& atoms,
        const double       globalSmF
    )
    {
        for (auto& atom : atoms)
        {
            const auto force = atom->getForce();
            atom->addForceOuter(force * globalSmF);
        }

        _simulationBox->resetForces();
    }

    /**
     * @brief Scale forces of smoothing zone molecules by their smoothing factor
     * for inner (QM) contribution
     *
     * @details This function iterates through all molecules in the smoothing
     * region and scales their atomic forces by their individual smoothing
     * factor (smF). This scaling is used in the hotspot smoothing algorithm
     * to weight the QM contribution of smoothing molecules. The smoothing
     * factor represents the degree to which a molecule should be treated with
     * QM methods, with smF = 1 being fully QM and smF = 0 being fully MM.
     */
    void QMMMMDEngine::scaleSmoothingMoleculeForcesInner()
    {
        for (auto& mol : _simulationBox->getMoleculesInsideZone(SMOOTHING))
        {
            const auto smF = mol.getSmoothingFactor();
            for (auto& atom : mol.getAtoms()) atom->scaleForce(smF);
        }
    }

    /**
     * @brief Scale forces of smoothing zone molecules by complementary
     * smoothing factor for outer (MM) contribution
     *
     * @details This function iterates through all molecules in the smoothing
     * region and scales their atomic forces by (1 - smoothing factor). This
     * complementary scaling is used in the hotspot smoothing algorithm to
     * weight the MM contribution of smoothing molecules. Since the total weight
     * must sum to 1, molecules with high QM character (high smF) receive low MM
     * weight (1 - smF), ensuring a smooth transition between QM and MM regions.
     */
    void QMMMMDEngine::scaleSmoothingMoleculeForcesOuter()
    {
        for (auto& mol : _simulationBox->getMoleculesInsideZone(SMOOTHING))
        {
            const auto smF = mol.getSmoothingFactor();
            for (auto& atom : mol.getAtoms()) atom->scaleForce(1 - smF);
        }
    }

}   // namespace engine