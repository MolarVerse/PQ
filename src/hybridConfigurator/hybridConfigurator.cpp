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

#include "hybridConfigurator.hpp"

#include <limits>          // for numeric_limits
#include <unordered_set>   // for unordered_set

#include "atom.hpp"             // for Atom
#include "exceptions.hpp"       // for HybridConfiguratorException
#include "hybridSettings.hpp"   // for settings::HybridSettings
#include "molecule.hpp"
#include "simulationBox.hpp"

namespace configurator
{

    /**
     * @brief Calculate the center of mass of the inner region center atoms
     *
     * @param simulationBox The simulation box containing all atoms
     *
     * @details This function calculates the mass-weighted center of the
     * atoms specified by the inner region center atom indices. The calculated
     * center is stored as the inner region center for the hybrid calculation.
     *
     * @throw HybridConfiguratorException if no center atoms are specified
     * (empty indices list)
     */
    void HybridConfigurator::calculateInnerRegionCenter(
        molsys::SimulationBox& simulationBox
    )
    {
        const auto& indices = simulationBox.getInnerRegionCenterAtomIndices();

        if (indices.empty())
        {
            throw exc::HybridConfiguratorException(
                "Cannot calculate inner region center: no center atoms "
                "specified"
            );
        }

        linalg::Vec3D center     = {0.0, 0.0, 0.0};
        double        total_mass = 0.0;
        const auto    positionAtom1 =
            simulationBox.getAtom(indices.at(0)).getPosition();

        for (const auto index : indices)
        {
            const auto& atom     = simulationBox.getAtom(index);
            const auto  mass     = atom.getMass();
            const auto  position = atom.getPosition();
            const auto  deltaPos = position - positionAtom1;

            center +=
                mass * (position - simulationBox.calcShiftVector(deltaPos));
            total_mass += mass;
        }

        center             /= total_mass;
        _innerRegionCenter  = center - simulationBox.calcShiftVector(center);
    }   // TODO: https://github.com/MolarVerse/PQ/issues/196

    /**
     * @brief Shift all atoms so that the inner region center is at the origin
     *
     * @param simulationBox The simulation box containing all atoms to be
     * shifted
     *
     * @details This function translates all atoms in the simulation box by
     * subtracting the inner region center position from each atom's
     * coordinates. After shifting, periodic boundary conditions are applied to
     * ensure atoms remain within the simulation box bounds.
     *
     * @note The inner region center must be calculated before calling this
     * function
     */
    void HybridConfigurator::shiftAtomsToInnerRegionCenter(
        molsys::SimulationBox& simulationBox
    )
    {
        for (auto& atom : simulationBox.getAtoms())
        {
            auto position = atom->getPosition() - _innerRegionCenter;
            simulationBox.applyPBC(position);
            atom->setPosition(position);
        }
    }

    /**
     * @brief Shift all atoms back to their original positions before centering
     *
     * @param simulationBox The simulation box containing all atoms to be
     * shifted back
     *
     * @details This function reverses the translation applied by
     *          shiftAtomsToInnerRegionCenter() by adding the inner region
     * center position back to each atom's coordinates. After shifting, periodic
     *          boundary conditions are applied to ensure atoms remain within
     * the simulation box bounds.
     *
     * @note This function should be called after
     * shiftAtomsToInnerRegionCenter() to restore the original atomic positions
     */
    void HybridConfigurator::shiftAtomsBackToInitialPositions(
        molsys::SimulationBox& simulationBox
    )
    {
        for (auto& atom : simulationBox.getAtoms())
        {
            auto position = atom->getPosition() + _innerRegionCenter;
            simulationBox.applyPBC(position);
            atom->setPosition(position);
        }
    }

    /**
     * @brief Assign hybrid zones to all molecules based on their distance from
     * the inner region center
     *
     * @param simulationBox The simulation box containing molecules to be
     * assigned to zones
     *
     * @details This function assigns each molecule in the simulation box to one
     * of four hybrid zones based on the distance of the molecule's center of
     * mass from the inner region center:
     *
     * - **CORE**: Distance ≤ core radius
     * - **LAYER**: core radius < distance ≤ (layer radius - smoothing region
     * thickness)
     * - **SMOOTHING**: (layer radius - smoothing region thickness) < distance ≤
     * layer radius
     * - **POINT_CHARGE** layer radius < distance ≤ (layer radius +
     * pointChargeThickness)
     * - **OUTER**: Distance > layer radius + pointChargeThickness
     *
     * The function calculates the center of mass for each molecule and
     * determines its zone assignment using the hybrid settings parameters (core
     * radius, layer radius, smoothing region thickness and point charge
     * thickness).
     *
     * @note The inner region center should be set to the origin (via
     *       shiftAtomsToInnerRegionCenter) before calling this function for
     *       accurate distance calculations
     */
    void HybridConfigurator::assignHybridZones(
        molsys::SimulationBox& simulationBox
    )
    {
        const auto coreRadius  = settings::HybridSettings::getCoreRadius();
        const auto layerRadius = settings::HybridSettings::getLayerRadius();
        const auto smoothingRegionThickness =
            settings::HybridSettings::getSmoothingRegionThickness();
        const auto pointChargeThickness =
            settings::HybridSettings::getPointChargeThickness();
        const auto coreEpsilon = std::numeric_limits<double>::epsilon();
        const bool coreEnabled = coreRadius > coreEpsilon;

        _molChangedZone = false;

        // Helper lambda to set zone and track changes
        auto setZone = [](auto& mol, auto newZone)
        {
            if (mol.getHybridZone() != newZone)
            {
                _molChangedZone = true;
                mol.setHybridZone(newZone);
            }
        };

        for (auto& mol : simulationBox.getMolecules())
        {
            mol.calculateCenterOfMass(simulationBox.getBox());

            if (mol.isForcedCore())
            {
                setZone(mol, molsys::HybridZone::CORE);
                continue;
            }

            if (mol.isForcedLayer())
            {
                setZone(mol, molsys::HybridZone::LAYER);
                continue;
            }

            const auto com = norm(mol.getCenterOfMass());

            if (mol.isForcedOuter())
            {
                if (com <= layerRadius + pointChargeThickness)
                    setZone(mol, molsys::HybridZone::POINT_CHARGE);
                else
                    setZone(mol, molsys::HybridZone::OUTER);

                continue;
            }

            if (coreEnabled && com <= coreRadius)
                setZone(mol, molsys::HybridZone::CORE);
            else if (com <= (layerRadius - smoothingRegionThickness))
                setZone(mol, molsys::HybridZone::LAYER);
            else if (com <= layerRadius)
                setZone(mol, molsys::HybridZone::SMOOTHING);
            else if (com <= layerRadius + pointChargeThickness)
                setZone(mol, molsys::HybridZone::POINT_CHARGE);
            else
                setZone(mol, molsys::HybridZone::OUTER);
        }
    }

    /**
     * @brief Activate all molecules in the simulation box
     *
     * @param simulationBox The simulation box containing molecules to be
     * activated
     *
     * @details This function activates all molecules regardless of their hybrid
     * zone assignment. This is typically used to reset the activation state
     * before applying selective activation/deactivation patterns.
     */
    void HybridConfigurator::activateMolecules(
        molsys::SimulationBox& simulationBox
    )
    {
        for (auto& mol : simulationBox.getMolecules()) mol.activateMolecule();
    }

    /**
     * @brief Deactivate molecules in the outer regions (POINT_CHARGE, OUTER)
     *
     * @param simulationBox The simulation box containing molecules to be
     * deactivated
     *
     * @details This function deactivates molecules in the outer hybrid zones:
     * POINT_CHARGE and OUTER regions. This is typically used during inner
     * region calculations where only the inner molecules should be active.
     */
    void HybridConfigurator::deactivateOuterMolecules(

        molsys::SimulationBox& simulationBox

    )
    {
        for (auto& mol : simulationBox.getMolecules())
        {
            const auto zone = mol.getHybridZone();

            if (zone == molsys::HybridZone::POINT_CHARGE ||
                zone == molsys::HybridZone::OUTER)
                mol.deactivateMolecule();
        }
    }

    /**
     * @brief Activate molecules within hybrid zone SMOOTHING
     *
     * @param simulationBox The simulation box containing the molecules
     */
    void HybridConfigurator::activateSmoothingMolecules(
        molsys::SimulationBox& simulationBox
    )
    {
        for (auto& mol : simulationBox.getMoleculesInsideZone(
                 molsys::HybridZone::SMOOTHING
             ))
            mol.activateMolecule();
    }

    /**
     * @brief Deactivate specific smoothing molecules by their indices
     *
     * @param inactiveMolecules Set of smoothing molecule indices (0-based
     * within smoothing zone) to be deactivated
     * @param simulationBox The simulation box containing the molecules
     *
     * @details This function deactivates only the smoothing molecules specified
     * in the inactiveMolecules set. The indices refer to the position within
     * the smoothing zone, not the global molecule index.
     */
    void HybridConfigurator::deactivateSmoothingMolecules(
        const std::unordered_set<size_t>& inactiveMolecules,
        molsys::SimulationBox&            simulationBox
    )
    {
        size_t count{0};
        for (auto& mol : simulationBox.getMoleculesInsideZone(
                 molsys::HybridZone::SMOOTHING
             ))
        {
            if (inactiveMolecules.contains(count))
                mol.deactivateMolecule();

            ++count;
        }
    }

    /**
     * @brief Toggle the activation state of all molecules in the simulation box
     *
     * @param simulationBox The simulation box containing molecules with their
     * activation state to be toggled
     *
     * @details This function toggles the activation state of each molecule in
     * the simulation box: active molecules are deactivated and inactive
     * molecules are activated. This operation is performed regardless of the
     * molecules' hybrid zone assignments and is useful for implementing
     * complementary calculations.
     */
    void HybridConfigurator::toggleMoleculeActivation(
        molsys::SimulationBox& simulationBox
    )
    {
        for (auto& mol : simulationBox.getMolecules())
        {
            if (mol.isActive())
                mol.deactivateMolecule();
            else
                mol.activateMolecule();
        }
    }

    /**
     * @brief Calculate smoothing factors for molecules in the smoothing region
     *
     * This function computes and assigns a smoothing factor to each molecule in
     * the smoothing region of the simulation box. The smoothing factor is
     * calculated based on the molecule's center of mass distance from the layer
     * radius, normalized by the smoothing region thickness. The formula used
     * ensures a smooth transition of the factor within the region.
     *
     * @param simulationBox Simulation box containing the molecules
     *
     * @throw HybridConfiguratorException if a molecule is outside the smoothing
     * region
     */
    void HybridConfigurator::calculateSmoothingFactors(
        molsys::SimulationBox& simulationBox
    )
    {
        const auto layer = settings::HybridSettings::getLayerRadius();
        const auto thickness =
            settings::HybridSettings::getSmoothingRegionThickness();

        for (auto& mol : simulationBox.getMoleculesInsideZone(
                 molsys::HybridZone::SMOOTHING
             ))
        {
            mol.calculateCenterOfMass(simulationBox.getBox());
            const auto com = norm(mol.getCenterOfMass());

            auto distanceFactor = (com - (layer - thickness)) / thickness;

            if (distanceFactor < 0.0 || distanceFactor > 1)
            {
                throw exc::HybridConfiguratorException(
                    "Cannot calculate smoothing factor for molecule outside "
                    "the "
                    "smoothing region"
                );
            }

            distanceFactor       -= 0.5;
            const auto dfSquared  = distanceFactor * distanceFactor;
            const auto smF        = (distanceFactor *
                              (dfSquared * (-6.0 * dfSquared + 5.0) - 1.875)) +
                             0.5;

            mol.setSmoothingFactor(smF);
        }
    }

    /********************************
     * standard getters and setters *
     ********************************/

    /**
     * @brief get the inner region center coordinates
     *
     * @return pq::Vec3D innerRegionCenter
     */
    linalg::Vec3D HybridConfigurator::getInnerRegionCenter() const
    {
        return _innerRegionCenter;
    }

    /** @brief get if any molecule changed its hybrid zone since last
     * assignation
     *
     * @return bool molChangedZone
     */
    bool HybridConfigurator::getMoleculeChangedZone()
    {
        return _molChangedZone;
    }

}   // namespace configurator
