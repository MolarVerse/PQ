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

#include <cstddef>
#include <cstdlib>
#include <format>
#include <functional>
#include <limits>
#include <vector>

#include "celllist.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "hybridSettings.hpp"
#include "intraNonBondedMap.hpp"
#include "molecule.hpp"
#include "vector3d.hpp"
#include "virial.hpp"

namespace engine
{
    namespace
    {
        /**
         * @brief Generate distance-based, unnormalized molecule weights.
         *
         * @param smoothingMol The smoothing molecule whose force deficit is
         * being redistributed.
         * @param recipientMolecules Recipient CORE/LAYER molecules.
         *
         * @return A vector with one switched-polynomial weight per recipient
         * molecule.
         *
         * @details Weights are computed from center-of-mass distance using
         * weightingRadius = 2 * HybridSettings::getLayerRadius(). For x = d/R
         * and x < 1, the switch is S(x)=1-10x^3+15x^4-6x^5; otherwise the
         * weight is 0. The returned values are unnormalized and normalized by
         * the caller.
         */
        std::vector<double> getDistanceWeights(
            const molsys::Molecule &smoothingMol,
            const std::vector<std::reference_wrapper<molsys::Molecule>>
                &recipientMolecules
        )
        {
            const auto weightingRadius =
                2.0 * settings::HybridSettings::getLayerRadius();
            std::vector<double> weights;
            weights.reserve(recipientMolecules.size());

            const auto       smoothingCOM = smoothingMol.getCenterOfMass();
            constexpr double x3Coeff      = 10.0;
            constexpr double x4Coeff      = 15.0;
            constexpr double x5Coeff      = 6.0;

            for (const auto &recipientMol : recipientMolecules)
            {
                const auto delta =
                    recipientMol.get().getCenterOfMass() - smoothingCOM;

                const auto distance = linalg::norm(delta);

                auto switchedWeight = 0.0;
                if (weightingRadius > 0.0)
                {
                    const auto x = distance / weightingRadius;

                    if (x < 1)
                    {
                        const auto x_3 = x * x * x;
                        const auto x_4 = x_3 * x;
                        const auto x_5 = x_4 * x;

                        switchedWeight =
                            1 - x3Coeff * x_3 + x4Coeff * x_4 - x5Coeff * x_5;
                    }
                }

                weights.push_back(switchedWeight);
            }

            return weights;
        }
    }   // namespace

    /**
     * @brief calculate QM/MM forces
     *
     */
    void QMMMMDEngine::calculateForces()
    {
        _getHybridConfigurator().calculateInnerRegionCenter(getSimulationBox());
        _getHybridConfigurator().shiftAtomsToInnerRegionCenter(
            getSimulationBox()
        );
        configurator::HybridConfigurator::assignHybridZones(getSimulationBox());
        _moltypeCheck();
        configurator::HybridConfigurator::calculateSmoothingFactors(
            getSimulationBox()
        );
        getCellList().updateCellList(getSimulationBox());
        getPhysicalDataPtr()->setNumberOfSmoothingMolecules(
            static_cast<double>(std::ranges::distance(
                getSimulationBox().getMoleculesInsideZone(
                    molsys::HybridZone::SMOOTHING
                )
            ))
        );

        _applySmoothing();

        _combineInnerOuterForces();
        _getHybridConfigurator().shiftAtomsBackToInitialPositions(
            getSimulationBox()
        );
    }

    /**
     * @brief Apply the selected smoothing method for QM/MM boundary
     * treatment
     *
     * @throws HybridMDEngineException if an unknown smoothing method is
     * requested
     *
     * @details This function dispatches to the appropriate smoothing
     * algorithm (hotspot or exact) based on the user-configured smoothing
     * method setting. The smoothing method determines how forces and
     * energies are calculated in the boundary region between QM and MM
     * zones. This is a wrapper function that delegates to either
     * applyHotspotSmoothing() or applyExactSmoothing().
     */
    void QMMMMDEngine::_applySmoothing()
    {
        using enum SmoothingMethod;

        const auto &smoothingMethod =
            settings::HybridSettings::getSmoothingMethod();

        switch (smoothingMethod)
        {
            case HOTSPOT: _applyHotspotSmoothing(); break;
            case EXACT: _applyExactSmoothing(); break;
        }
    }

    /**
     * @brief Apply exact smoothing algorithm for QM/MM boundary treatment
     *
     * @details This function implements the exact smoothing algorithm by
     * iterating through all 2^n combinations of smoothing molecules being
     * active/inactive in the inner (QM) region.
     *
     * @note Computational cost: O(2^n) where n = number of smoothing
     * molecules
     */
    void QMMMMDEngine::_applyExactSmoothing()
    {
        using enum molsys::Periodicity;
        using std::ranges::distance;

        linalg::tensor3D virial     = {0.0};
        auto             numQMAtoms = 0.0;
        auto            &atoms      = getSimulationBox().getAtoms();
        const auto       nSmMol     = static_cast<size_t>(distance(
            getSimulationBox().getMoleculesInsideZone(
                molsys::HybridZone::SMOOTHING
            )
        ));

        // Loop over all combinations of smoothing molecules
        for (size_t i = 0; i < (1U << nSmMol); ++i)
        {
            // STEP 1: Generate set of inactive molecules and calculate
            // associated global smoothing factor for this configuration
            const auto inactiveSmMol =
                _generateInactiveSmoothingMoleculeSet(i, nSmMol);
            const auto globalSmF =
                _calculateGlobalSmoothingFactor(inactiveSmMol);

            // STEP 2: Setup and run QM calculation, accumulate QM forces
            // and QM virial contribution and the number of QM atoms for
            // this combination
            configurator::HybridConfigurator::activateMolecules(
                getSimulationBox()
            );
            configurator::HybridConfigurator::deactivateOuterMolecules(
                getSimulationBox()
            );
            configurator::HybridConfigurator::deactivateSmoothingMolecules(
                inactiveSmMol,
                getSimulationBox()
            );

            numQMAtoms +=
                static_cast<double>(getSimulationBox().getNumberOfQMAtoms()) *
                globalSmF;

            // to not carry over qm_charges of atoms which are not qm
            // anymore
            getSimulationBox().resetQMCharges();

            getQMRunner()
                ->run(getSimulationBox(), getPhysicalData(), NON_PERIODIC);

            virial += virial::calculateQMVirial(getSimulationBox()) * globalSmF;
            virial +=
                virial::intraMolecularVirialCorrection(getSimulationBox()) *
                globalSmF;
            _addScaledCurrentForcesToInnerAndReset(atoms, globalSmF);

            // STEP 3: Setup and run MM calculation, accumulate MM forces
            // and MM virial contribution
            configurator::HybridConfigurator::toggleMoleculeActivation(
                getSimulationBox()
            );

            if (settings::GeneralSettings::isCellListActivated())
            {
                getCellList().assignMoleculeHybridZoneIndices();
                getCellList().assignWaterMoleculeIndices(
                    getSimulationBox(),
                    getSettings().waterModelSettings
                );
            }

            getPotential()->calculateQMMMForces(
                getSimulationBox(),
                getPhysicalData(),
                getCellList(),
                getSettings().waterModelSettings
            );

            _getInterWater().calculateQMMMForces(
                getSimulationBox(),
                getPhysicalData(),
                getPotential()->getCoulombPotSharedPtr(),
                getCellList()
            );

            getIntraNonBonded()->calculate(
                getSimulationBox(),
                getPhysicalData()
            );

            virial += virial::calculateVirial(getSimulationBox()) * globalSmF;
            virial +=
                virial::intraMolecularVirialCorrection(getSimulationBox()) *
                globalSmF;
            _addScaledCurrentForcesToOuterAndReset(atoms, globalSmF);

            // bonded interactions directly add to physical data virial
            getPhysicalDataPtr()->setVirial({0.0});

            getForceField()->calculateBondedInteractions(
                getSimulationBox(),
                getPhysicalData()
            );

            _getIntraWater().calculate(getSimulationBox(), getPhysicalData());

            virial += getPhysicalDataPtr()->getVirial() * globalSmF;
            virial +=
                virial::intraMolecularVirialCorrection(getSimulationBox()) *
                globalSmF;
            _addScaledCurrentForcesToOuterAndReset(atoms, globalSmF);

            // STEP 4: Scale and accumulate hybrid energies and delete temp
            // files --> following configs cannot continue if the QM calc
            // fails
            _scaleAndAccumulateEnergies(globalSmF);
            getPhysicalDataPtr()->resetEnergies();
            deleteTmpFiles();
        }

        // STEP 5: Set energies, virial and numQMAtoms to accumulated values
        _moveEnergiesToPhysicalData();
        getPhysicalDataPtr()->setVirial(virial);
        getPhysicalDataPtr()->setNumberOfQMAtoms(numQMAtoms);
    }

    /**
     * @brief Apply hotspot smoothing algorithm for QM/MM boundary treatment
     *
     * @details This function implements the hotspot smoothing algorithm by
     * running separate QM and MM calculations and scaling forces of
     * smoothing molecules according to their individual smoothing factors.
     * More computationally efficient than exact smoothing but less
     * rigorous.
     *
     * @warning The energies yielded by this smoothing method are not
     * correct
     *
     * @note Computational cost: O(1) - constant time regardless of number
     * of smoothing molecules
     */
    void QMMMMDEngine::_applyHotspotSmoothing()
    {
        using enum molsys::Periodicity;

        auto            &atoms  = getSimulationBox().getAtoms();
        linalg::tensor3D virial = {0.0};

        // Set number of QM atoms in physical data for output purposes
        _setNumberOfQMAtoms();

        // STEP 1: Setup and run QM calculation, scale forces of smoothing
        // molecules with smF
        configurator::HybridConfigurator::activateMolecules(getSimulationBox());
        configurator::HybridConfigurator::deactivateOuterMolecules(
            getSimulationBox()
        );

        getQMRunner()->run(getSimulationBox(), getPhysicalData(), NON_PERIODIC);

        if (settings::HybridSettings::getQMForceDist() == QMForceDist::NONE)
            _scaleSmoothingMoleculeForcesInner();
        else
            _distributeSmoothingMolQMForces();

        virial += virial::calculateQMVirial(getSimulationBox());
        virial += virial::intraMolecularVirialCorrection(getSimulationBox());
        _addCurrentForcesToInnerAndReset(atoms);

        // STEP 2: Setup and run inter-nonbonded calculation between
        // MM-MM , CORE-MM , LAYER+SMOOTHING-MM and scale forces of
        // smoothing molecules with smF
        configurator::HybridConfigurator::toggleMoleculeActivation(
            getSimulationBox()
        );

        if (settings::GeneralSettings::isCellListActivated())
        {
            getCellList().assignMoleculeHybridZoneIndices();
            getCellList().assignWaterMoleculeIndices(
                getSimulationBox(),
                getSettings().waterModelSettings
            );
        }

        getPotential()->calculateQMMMForces(
            getSimulationBox(),
            getPhysicalData(),
            getCellList(),
            getSettings().waterModelSettings
        );

        _getInterWater().calculateQMMMForces(
            getSimulationBox(),
            getPhysicalData(),
            getPotential()->getCoulombPotSharedPtr(),
            getCellList()
        );

        _scaleSmoothingMoleculeForcesInner();
        virial += virial::calculateVirial(getSimulationBox());
        virial += virial::intraMolecularVirialCorrection(getSimulationBox());
        _addCurrentForcesToOuterAndReset(atoms);

        // STEP 3: Calculate inter-nonbonded forces between SMOOTHING
        // molecules and scale forces of smoothing molecules with (1 - smF)

        getPotential()->calculateHotspotSmoothingMMForces(
            getSimulationBox(),
            getPhysicalData(),
            getCellList(),
            getSettings().waterModelSettings
        );

        _getInterWater().calculateHotspotSmoothingMMForces(
            getSimulationBox(),
            getPhysicalData(),
            getPotential()->getCoulombPotSharedPtr(),
            getCellList()
        );

        _scaleSmoothingMoleculeForcesOuter();
        virial += virial::calculateVirial(getSimulationBox());
        virial += virial::intraMolecularVirialCorrection(getSimulationBox());
        _addCurrentForcesToOuterAndReset(atoms);

        // STEP 4: Setup and run intra-nonbonded calculation and scale
        // forces of smoothing molecules with (1 - smF)

        configurator::HybridConfigurator::activateSmoothingMolecules(
            getSimulationBox()
        );

        getIntraNonBonded()->calculate(getSimulationBox(), getPhysicalData());

        _scaleSmoothingMoleculeForcesOuter();
        virial += virial::calculateVirial(getSimulationBox());
        virial += virial::intraMolecularVirialCorrection(getSimulationBox());
        _addCurrentForcesToOuterAndReset(atoms);

        // STEP 5: Run intra-bonded calculation and scale forces of
        // smoothing molecules with (1 - smF)

        // bonded interactions directly add to physical data virial
        getPhysicalDataPtr()->setVirial({0.0});

        getForceField()->calculateBondedInteractions(
            getSimulationBox(),
            getPhysicalData()
        );

        _getIntraWater().calculate(getSimulationBox(), getPhysicalData());

        _scaleSmoothingMoleculeForcesOuter();
        virial += getPhysicalDataPtr()->getVirial();
        virial += virial::intraMolecularVirialCorrection(getSimulationBox());
        _addCurrentForcesToOuterAndReset(atoms);

        getPhysicalDataPtr()->setVirial(virial);
    }

    /**
     * @brief Set the number of QM atoms in physical data for output
     * purposes
     *
     * @details This function temporarily configures the simulation box to
     * count only QM atoms by activating all molecules and then deactivating
     * outer molecules. The count is stored in physical data and then all
     * molecules are reactivated to restore the original state.
     */
    void QMMMMDEngine::_setNumberOfQMAtoms()
    {
        configurator::HybridConfigurator::activateMolecules(getSimulationBox());
        configurator::HybridConfigurator::deactivateOuterMolecules(
            getSimulationBox()
        );
        const auto nQMAtoms = getSimulationBox().getNumberOfQMAtoms();
        getPhysicalDataPtr()->setNumberOfQMAtoms(static_cast<double>(nQMAtoms));
        configurator::HybridConfigurator::activateMolecules(getSimulationBox());
    }

    /**
     * @brief Validates that all non-core molecules have a non-zero moltype
     *
     * @details This function iterates through all molecules and verifies
     * that each molecule outside the QM core has a moltype that is not
     * zero.
     *
     * @throws HybridMDEngineException if any non-core molecule has moltype
     * == 0
     */
    void QMMMMDEngine::_moltypeCheck()
    {
        size_t count = 0;
        for (const auto &mol : getSimulationBox().getMolecules())
        {
            if (mol.getMoltype() == MolType{0} &&
                mol.getHybridZone() != molsys::HybridZone::CORE)
            {
                throw exc::HybridMDEngineException(
                    std::format(
                        "Molecule number {} is outside the QM core and has "
                        "moltype 0. All molecules outside the QM core must "
                        "have a non-zero moltype assigned.",
                        count
                    )
                );
            }
            ++count;
        }
    }

    /**
     * @brief Scale current PhysicalData energies and add them to the
     * internal QM/MM PhyscialData object. Used for the exact smoothing
     * method.
     *
     * @param globalSmF Global smoothing factor for the current
     * configuration.
     */
    void QMMMMDEngine::_scaleAndAccumulateEnergies(double globalSmF)
    {
        _qmmmPhysicalData.addQMEnergy(
            getPhysicalDataPtr()->getQMEnergy() * globalSmF
        );
        _qmmmPhysicalData.addCoulombEnergy(
            getPhysicalDataPtr()->getCoulombEnergy() * globalSmF
        );
        _qmmmPhysicalData.addNonCoulombEnergy(
            getPhysicalDataPtr()->getNonCoulombEnergy() * globalSmF
        );
        _qmmmPhysicalData.addBondEnergy(
            getPhysicalDataPtr()->getBondEnergy() * globalSmF
        );
        _qmmmPhysicalData.addAngleEnergy(
            getPhysicalDataPtr()->getAngleEnergy() * globalSmF
        );
        _qmmmPhysicalData.addDihedralEnergy(
            getPhysicalDataPtr()->getDihedralEnergy() * globalSmF
        );
        _qmmmPhysicalData.addImproperEnergy(
            getPhysicalDataPtr()->getImproperEnergy() * globalSmF
        );
        _qmmmPhysicalData.setIntraCoulombEnergy(
            _qmmmPhysicalData.getIntraCoulombEnergy() +
            (getPhysicalDataPtr()->getIntraCoulombEnergy() * globalSmF)
        );
        _qmmmPhysicalData.setIntraNonCoulombEnergy(
            _qmmmPhysicalData.getIntraNonCoulombEnergy() +
            (getPhysicalDataPtr()->getIntraNonCoulombEnergy() * globalSmF)
        );
    }

    /**
     * @brief Transfer internal accumulated energies from the QM/MM
     * PhysicalData object to PhysicalData and reset the internal object.
     * Used for the exact smoothing method.
     */
    void QMMMMDEngine::_moveEnergiesToPhysicalData()
    {
        getPhysicalDataPtr()->setQMEnergy(_qmmmPhysicalData.getQMEnergy());
        getPhysicalDataPtr()->setCoulombEnergy(
            _qmmmPhysicalData.getCoulombEnergy()
        );
        getPhysicalDataPtr()->setNonCoulombEnergy(
            _qmmmPhysicalData.getNonCoulombEnergy()
        );
        getPhysicalDataPtr()->setBondEnergy(_qmmmPhysicalData.getBondEnergy());
        getPhysicalDataPtr()->setAngleEnergy(_qmmmPhysicalData.getAngleEnergy()
        );
        getPhysicalDataPtr()->setDihedralEnergy(
            _qmmmPhysicalData.getDihedralEnergy()
        );
        getPhysicalDataPtr()->setImproperEnergy(
            _qmmmPhysicalData.getImproperEnergy()
        );
        getPhysicalDataPtr()->setIntraCoulombEnergy(
            _qmmmPhysicalData.getIntraCoulombEnergy()
        );
        getPhysicalDataPtr()->setIntraNonCoulombEnergy(
            _qmmmPhysicalData.getIntraNonCoulombEnergy()
        );

        _qmmmPhysicalData.reset();
    }

    /**
     * @brief Dispatch QM-force redistribution for smoothing-zone molecules.
     *
     * @details Selects the redistribution strategy configured via
     * HybridSettings::getQMForceDist().
     * - NONE: scale smoothing-zone forces by smF only.
     * - EQUAL: redistribute each smoothing-molecule deficit equally to
     *   CORE/LAYER molecules, then split by atomic mass within each
     * molecule.
     * - RANDOM: redistribute each smoothing-molecule deficit randomly to
     *   CORE/LAYER molecules, then split by atomic mass within each
     * molecule.
     * - DISTANCE_WEIGHTED: redistribute each smoothing-molecule deficit
     * using switched-polynomial COM distance weights, then split by atomic
     * mass within each recipient molecule.
     *
     * @note For each smoothing molecule, the current atom forces are first
     * scaled by smF and only the force deficit F*(1-smF) is redistributed.
     *
     * @throws HybridMDEngineException if no CORE/LAYER recipient molecules
     * are available for redistribution.
     */
    void QMMMMDEngine::_distributeSmoothingMolQMForces()
    {
        const auto type = settings::HybridSettings::getQMForceDist();
        std::vector<std::reference_wrapper<molsys::Molecule>>
            recipientMolecules;

        for (auto &mol : getSimulationBox().getMoleculesInsideZone(
                 molsys::HybridZone::CORE
             ))
            recipientMolecules.emplace_back(mol);

        for (auto &mol : getSimulationBox().getMoleculesInsideZone(
                 molsys::HybridZone::LAYER
             ))
            recipientMolecules.emplace_back(mol);

        if (recipientMolecules.empty())
        {
            throw exc::HybridMDEngineException(
                "Cannot redistribute smoothing inner force: no CORE/LAYER "
                "molecules available."
            );
        }

        for (auto &smoothingMol : getSimulationBox().getMoleculesInsideZone(
                 molsys::HybridZone::SMOOTHING
             ))
        {
            const auto smF = smoothingMol.getSmoothingFactor();

            auto deficitForce = linalg::Vec3D{0.0};
            for (auto &atom : smoothingMol.getAtoms())
            {
                deficitForce += atom->getForce() * (1 - smF);
                atom->scaleForce(smF);
            }

            auto weights = std::vector<double>(recipientMolecules.size(), 0.0);

            switch (type)
            {
                using enum QMForceDist;

                case NONE: continue;
                case EQUAL:
                    weights = std::vector<double>(recipientMolecules.size(), 1);
                    break;
                case RANDOM:
                    weights = _getRandomWeights(recipientMolecules);
                    break;
                case DISTANCE_WEIGHTED:
                    weights =
                        getDistanceWeights(smoothingMol, recipientMolecules);
                    break;
            }

            // normalize weights to sum to 1
            auto weightSum = 0.0;
            for (const auto weight : weights) weightSum += weight;

            if (std::abs(weightSum) <= std::numeric_limits<double>::epsilon())
            {
                // fallback: equal distribution
                const auto equal = 1.0 / static_cast<double>(weights.size());
                for (auto &weight : weights) weight = equal;
            }
            else
            {
                for (auto &weight : weights) weight /= weightSum;
            }

            for (size_t i = 0; i < recipientMolecules.size(); ++i)
            {
                const auto molShare     = deficitForce * weights[i];
                auto      &recipientMol = recipientMolecules[i].get();
                const auto molMass      = recipientMol.getMolMass();

                for (auto &atom : recipientMol.getAtoms())
                {
                    const auto massFraction = atom->getMass() / molMass;
                    atom->addForce(molShare * massFraction);
                }
            }
        }
    }

    /**
     * @brief Generate random, unnormalized molecule weights.
     *
     * @param recipientMolecules Recipient CORE/LAYER molecules.
     *
     * @return A vector with one random value in [0, 1] per recipient
     * molecule.
     *
     * @details The returned values are intentionally unnormalized. The
     * caller performs normalization and handles zero-sum fallback logic.
     */
    std::vector<double> QMMMMDEngine::_getRandomWeights(
        const std::vector<std::reference_wrapper<molsys::Molecule>>
            &recipientMolecules
    )
    {
        std::vector<double> randomWeights(recipientMolecules.size(), 0.0);

        for (auto &weight : randomWeights)
            weight = _rng.getUniformRealDistribution(0.0, 1);

        return randomWeights;
    }

}   // namespace engine
