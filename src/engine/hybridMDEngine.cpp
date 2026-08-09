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

#include "hybridMDEngine.hpp"

using enum simulationBox::HybridZone;

namespace engine
{
    /**
     * @brief Combine inner (QM) and outer (MM) forces into final atomic forces
     *
     * @details This function finalizes the hybrid QM/MM force calculation by
     * summing the separately accumulated inner and outer force contributions
     * for each atom. After the smoothing algorithm has computed and accumulated
     * QM forces (inner) and MM forces (outer) with appropriate weighting, this
     * function combines them to produce the final total force on each atom that
     * will be used for integration.
     */
    void HybridMDEngine::combineInnerOuterForces()
    {
        for (auto& atom : _simulationBox->getAtoms())
        {
            const auto innerForce = atom->getForceInner();
            const auto outerForce = atom->getForceOuter();
            atom->setForce(innerForce + outerForce);
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
    void HybridMDEngine::addCurrentForcesToInnerAndReset(
        pq::SharedAtomVec& atoms
    )
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
     * simulation box are reset to zero. This function is used in exact
     * smoothing where forces need to be weighted by the configuration-specific
     * global smoothing factor.
     */
    void HybridMDEngine::addScaledCurrentForcesToInnerAndReset(
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
    void HybridMDEngine::addCurrentForcesToOuterAndReset(
        pq::SharedAtomVec& atoms
    )
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
     * simulation box are reset to zero. This function is used in exact
     * smoothing where forces need to be weighted by the configuration-specific
     * global smoothing factor.
     */
    void HybridMDEngine::addScaledCurrentForcesToOuterAndReset(
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
     * for inner contribution
     *
     * @details This function iterates through all molecules in the smoothing
     * region and scales their atomic forces by their individual smoothing
     * factor (smF). This scaling is used in the hotspot smoothing algorithm
     * to weight the QM contribution of smoothing molecules. The smoothing
     * factor represents the degree to which a molecule should be treated with
     * QM methods, with smF = 1 being fully QM and smF = 0 being fully MM.
     */
    void HybridMDEngine::scaleSmoothingMoleculeForcesInner()
    {
        for (auto& mol : _simulationBox->getMoleculesInsideZone(SMOOTHING))
        {
            const auto smF = mol.getSmoothingFactor();
            for (auto& atom : mol.getAtoms()) atom->scaleForce(smF);
        }
    }

    /**
     * @brief Scale forces of smoothing zone molecules by complementary
     * smoothing factor for outer contribution
     *
     * @details This function iterates through all molecules in the smoothing
     * region and scales their atomic forces by (1 - smoothing factor). This
     * complementary scaling is used in the hotspot smoothing algorithm to
     * weight the MM contribution of smoothing molecules. Since the total weight
     * must sum to 1, molecules with high QM character (high smF) receive low MM
     * weight (1 - smF), ensuring a smooth transition between QM and MM regions.
     */
    void HybridMDEngine::scaleSmoothingMoleculeForcesOuter()
    {
        for (auto& mol : _simulationBox->getMoleculesInsideZone(SMOOTHING))
        {
            const auto smF = mol.getSmoothingFactor();
            for (auto& atom : mol.getAtoms()) atom->scaleForce(1 - smF);
        }
    }

    /**
     * @brief Generate set of inactive smoothing molecule indices from bit
     * pattern used in the "exact" smoothing algorithm
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
    std::unordered_set<size_t> HybridMDEngine::
        generateInactiveSmoothingMoleculeSet(
            size_t bitPattern,
            size_t totalMolecules
        ) const
    {
        std::unordered_set<size_t> inactiveMolecules;

        for (size_t j = 0; j < totalMolecules; ++j)
            if (bitPattern & (1u << j))
                inactiveMolecules.insert(j);

        return inactiveMolecules;
    }

    /**
     * @brief Calculate the global smoothing factor for "exact" smoothing
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
    double HybridMDEngine::calculateGlobalSmoothingFactor(
        const std::unordered_set<size_t>& inactiveForInnerCalcMolecules
    ) const
    {
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