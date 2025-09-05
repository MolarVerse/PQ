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

#include <unordered_set>   // for unordered_set

#include "atom.hpp"             // for Atom
#include "exceptions.hpp"       // for HybridConfiguratorException
#include "hybridSettings.hpp"   // for HybridSettings
#include "simulationBox.hpp"    // for SimulationBox

using enum simulationBox::HybridZone;

using namespace pq;
using namespace configurator;
using namespace customException;
using namespace settings;
using namespace simulationBox;

/**
 * @brief Calculate the center of mass of the inner region center atoms
 *
 * @param simBox The simulation box containing all atoms
 *
 * @details This function calculates the mass-weighted center of the
 * atoms specified by the inner region center atom indices. The calculated
 * center is stored as the inner region center for the hybrid calculation.
 *
 * @throw HybridConfiguratorException if no center atoms are specified (empty
 * indices list)
 */
void HybridConfigurator::calculateInnerRegionCenter(SimBox& simBox)
{
    const auto& indices = simBox.getInnerRegionCenterAtomIndices();

    if (indices.empty())
        throw(HybridConfiguratorException(
            "Cannot calculate inner region center: no center atoms specified"
        ));

    Vec3D      center        = {0.0, 0.0, 0.0};
    double     total_mass    = 0.0;
    const auto positionAtom1 = simBox.getAtom(indices.at(0)).getPosition();

    for (const auto index : indices)
    {
        const auto& atom     = simBox.getAtom(index);
        const auto  mass     = atom.getMass();
        const auto  position = atom.getPosition();
        const auto  deltaPos = position - positionAtom1;

        center     += mass * (position - simBox.calcShiftVector(deltaPos));
        total_mass += mass;
    }

    center             /= total_mass;
    _innerRegionCenter  = center - simBox.calcShiftVector(center);
}   // TODO: https://github.com/MolarVerse/PQ/issues/196

/**
 * @brief Shift all atoms so that the inner region center is at the origin
 *
 * @param simBox The simulation box containing all atoms to be shifted
 *
 * @details This function translates all atoms in the simulation box by
 * subtracting the inner region center position from each atom's coordinates.
 * After shifting, periodic boundary conditions are applied to ensure atoms
 * remain within the simulation box bounds.
 *
 * @note The inner region center must be calculated before calling this function
 */
void HybridConfigurator::shiftAtomsToInnerRegionCenter(SimBox& simBox)
{
    for (auto& atom : simBox.getAtoms())
    {
        auto position = atom->getPosition() - _innerRegionCenter;
        simBox.applyPBC(position);
        atom->setPosition(position);
    }
}

/**
 * @brief Shift all atoms back to their original positions before centering
 *
 * @param simBox The simulation box containing all atoms to be shifted back
 *
 * @details This function reverses the translation applied by
 *          shiftAtomsToInnerRegionCenter() by adding the inner region center
 *          position back to each atom's coordinates. After shifting, periodic
 *          boundary conditions are applied to ensure atoms remain within the
 *          simulation box bounds.
 *
 * @note This function should be called after shiftAtomsToInnerRegionCenter()
 *       to restore the original atomic positions
 */
void HybridConfigurator::shiftAtomsBackToInitialPositions(SimBox& simBox)
{
    for (auto& atom : simBox.getAtoms())
    {
        auto position = atom->getPosition() + _innerRegionCenter;
        simBox.applyPBC(position);
        atom->setPosition(position);
    }
}

/**
 * @brief Assign hybrid zones to all molecules based on their distance from the
 * inner region center
 *
 * @param simBox The simulation box containing molecules to be assigned to zones
 *
 * @details This function assigns each molecule in the simulation box to one of
 * four hybrid zones based on the distance of the molecule's center of mass from
 * the inner region center:
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
 * The function calculates the center of mass for each molecule and determines
 * its zone assignment using the hybrid settings parameters (core radius, layer
 * radius, smoothing region thickness and point charge thickness).
 *
 * @note The inner region center should be set to the origin (via
 *       shiftAtomsToInnerRegionCenter) before calling this function for
 *       accurate distance calculations
 */
void HybridConfigurator::assignHybridZones(SimBox& simBox)
{
    const auto coreRadius  = HybridSettings::getCoreRadius();
    const auto layerRadius = HybridSettings::getLayerRadius();
    const auto smoothingRegionThickness =
        HybridSettings::getSmoothingRegionThickness();
    const auto pointChargeThickness = HybridSettings::getPointChargeThickness();

    // Helper lambda to set zone and track changes
    auto setZone = [this](auto& mol, HybridZone newZone)
    {
        if (mol.getHybridZone() != newZone)
        {
            _molChangedZone = true;
            mol.setHybridZone(newZone);
        }
    };

    for (auto& mol : simBox.getMolecules())
    {
        mol.calculateCenterOfMass(simBox.getBox());
        const auto com = norm(mol.getCenterOfMass());

        if (com <= coreRadius)
            setZone(mol, CORE);
        else if (com <= (layerRadius - smoothingRegionThickness))
            setZone(mol, LAYER);
        else if (com <= layerRadius)
        {
            setZone(mol, SMOOTHING);
            ++_numberSmoothingMolecules;
        }
        else if (com <= layerRadius + pointChargeThickness)
            setZone(mol, POINT_CHARGE);
        else
            setZone(mol, OUTER);
    }
}

/**
 * @brief Activate all molecules in the simulation box
 *
 * @param simBox The simulation box containing molecules to be activated
 *
 * @details This function activates all molecules regardless of their hybrid
 * zone assignment. This is typically used to reset the activation state before
 * applying selective activation/deactivation patterns.
 */
void HybridConfigurator::activateMolecules(pq::SimBox& simBox)
{
    for (auto& mol : simBox.getMolecules()) mol.activateMolecule();
}

/**
 * @brief Deactivate molecules in the inner regions (CORE, LAYER, SMOOTHING)
 *
 * @param simBox The simulation box containing molecules to be deactivated
 *
 * @details This function deactivates molecules in the inner hybrid zones:
 * CORE, LAYER, and SMOOTHING regions. This is typically used during outer
 * region calculations where only the outer molecules (POINT_CHARGE and OUTER)
 * should be active.
 */
void HybridConfigurator::deactivateInnerMolecules(pq::SimBox& simBox)
{
    for (auto& mol : simBox.getMolecules())
    {
        const auto zone = mol.getHybridZone();

        if (zone == CORE || zone == LAYER || zone == SMOOTHING)
            mol.deactivateMolecule();
    }
}

/**
 * @brief Deactivate molecules in the outer regions (POINT_CHARGE, OUTER)
 *
 * @param simBox The simulation box containing molecules to be deactivated
 *
 * @details This function deactivates molecules in the outer hybrid zones:
 * POINT_CHARGE and OUTER regions. This is typically used during inner
 * region calculations where only the inner molecules should be active.
 */
void HybridConfigurator::deactivateOuterMolecules(pq::SimBox& simBox)
{
    for (auto& mol : simBox.getMolecules())
    {
        const auto zone = mol.getHybridZone();

        if (zone == POINT_CHARGE || zone == OUTER)
            mol.deactivateMolecule();
    }
}

/**
 * @brief Activate specific smoothing molecules by their indices
 *
 * @param inactiveMolecules Set of smoothing molecule indices (0-based within
 * smoothing zone) to be activated
 * @param simBox The simulation box containing the molecules
 *
 * @details This function activates only the smoothing molecules specified
 * in the activeMolecules set. The indices refer to the position within
 * the smoothing zone, not the global molecule index.
 */
void HybridConfigurator::activateSmoothingMolecules(
    std::unordered_set<size_t> activeMolecules,
    pq::SimBox&                simBox
)
{
    size_t count{0};
    for (auto& mol : simBox.getMoleculesInsideZone(SMOOTHING))
    {
        if (activeMolecules.contains(count))
            mol.activateMolecule();

        ++count;
    }
}

/**
 * @brief Deactivate specific smoothing molecules by their indices
 *
 * @param inactiveMolecules Set of smoothing molecule indices (0-based within
 * smoothing zone) to be deactivated
 * @param simBox The simulation box containing the molecules
 *
 * @details This function deactivates only the smoothing molecules specified
 * in the inactiveMolecules set. The indices refer to the position within
 * the smoothing zone, not the global molecule index.
 */
void HybridConfigurator::deactivateSmoothingMolecules(
    std::unordered_set<size_t> inactiveMolecules,
    pq::SimBox&                simBox
)
{
    size_t count{0};
    for (auto& mol : simBox.getMoleculesInsideZone(SMOOTHING))
    {
        if (inactiveMolecules.contains(count))
            mol.deactivateMolecule();

        ++count;
    }
}

/**
 * @brief Calculate smoothing factors for molecules in the smoothing region
 *
 * This function computes and assigns a smoothing factor to each molecule in the
 * smoothing region of the simulation box. The smoothing factor is calculated
 * based on the molecule's center of mass distance from the layer radius,
 * normalized by the smoothing region thickness. The formula used ensures a
 * smooth transition of the factor within the region.
 *
 * @param simBox Simulation box containing the molecules
 *
 * @throw HybridConfiguratorException if a molecule is outside the smoothing
 * region
 */
void HybridConfigurator::calculateSmoothingFactors(pq::SimBox& simBox)
{
    const auto layer     = HybridSettings::getLayerRadius();
    const auto thickness = HybridSettings::getSmoothingRegionThickness();

    for (auto& mol : simBox.getMoleculesInsideZone(SMOOTHING))
    {
        mol.calculateCenterOfMass(simBox.getBox());
        const auto com = norm(mol.getCenterOfMass());

        const auto distanceFactor = (com - (layer - thickness)) / thickness;

        if (distanceFactor < 0.0 || distanceFactor > 1.0)
            throw(HybridConfiguratorException(
                "Cannot calculate smoothing factor for molecule outside the "
                "smoothing region"
            ));

        const auto dF  = distanceFactor - 0.5;
        const auto smF = dF * (dF * dF * (-6.0 * dF * dF + 5.0) - 1.875) + 0.5;

        mol.setSmoothingFactor(smF);
    }
}

/********************************
 * standard getters and setters *
 ********************************/

/**
 * @brief get the number of molecules in the point charge region of the
 * hybrid calculation
 *
 * @return size_t numberSmoothingMolecules
 */
size_t HybridConfigurator::getNumberSmoothingMolecules()
{
    return _numberSmoothingMolecules;
}

/**
 * @brief get the inner region center coordinates
 *
 * @return pq::Vec3D innerRegionCenter
 */
Vec3D HybridConfigurator::getInnerRegionCenter() const
{
    return _innerRegionCenter;
}

/** @brief get if a molecule changed its hybrid zone since last assignation
 *
 * @return bool molChangedZone
 */
bool HybridConfigurator::getMoleculeChangedZone() { return _molChangedZone; }

/**
 * @brief set the number of molecules in the point charge region of the
 * hybrid calculation
 *
 * @param count
 */
void HybridConfigurator::setNumberSmoothingMolecules(size_t count)
{
    _numberSmoothingMolecules = count;
}

/**
 * @brief set whether a molecule changed its hybrid zone since last assignation
 *
 * @param changed
 */
void HybridConfigurator::setMoleculeChangedZone(bool changed)
{
    _molChangedZone = changed;
}