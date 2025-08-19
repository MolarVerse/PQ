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

#include <stdexcept>       // for domain_error
#include <unordered_set>   // for unordered_set

#include "atom.hpp"             // for Atom
#include "hybridSettings.hpp"   // for HybridSettings
#include "simulationBox.hpp"    // for SimulationBox

using enum simulationBox::HybridZone;

using namespace pq;
using namespace configurator;
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
 * @throw std::domain_error if no center atoms are specified (empty indices
 * list)
 */
void HybridConfigurator::calculateInnerRegionCenter(SimBox& simBox)
{
    const auto& indices = simBox.getInnerRegionCenterAtomIndices();

    if (indices.empty())
        throw(std::domain_error(
            "Cannot calculate inner region center: no center atoms specified"
        ));

    Vec3D  center     = {0.0, 0.0, 0.0};
    double total_mass = 0.0;

    for (const auto index : indices)
    {
        const auto& atom  = simBox.getAtom(index);
        const auto  mass  = atom.getMass();
        center           += atom.getPosition() * mass;
        total_mass       += mass;
    }

    center /= total_mass;
    setInnerRegionCenter(center);
}

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
 * pointChargeRadius)
 * - **OUTER**: Distance > layer radius + pointChargeRadius
 *
 * The function calculates the center of mass for each molecule and determines
 * its zone assignment using the hybrid settings parameters (core radius, layer
 * radius, smoothing region thickness and point charge radius).
 *
 * @note The inner region center should be set to the origin (via
 *       shiftAtomsToInnerRegionCenter) before calling this function for
 *       accurate distance calculations
 */
void HybridConfigurator::assignHybridZones(SimBox& simBox)
{
    auto numberSmoothingMolecules = getNumberSmoothingMolecules();

    for (auto& mol : simBox.getMolecules())
    {
        mol.calculateCenterOfMass(simBox.getBox());
        const auto com = norm(mol.getCenterOfMass());

        const auto coreRadius  = HybridSettings::getCoreRadius();
        const auto layerRadius = HybridSettings::getLayerRadius();
        const auto smoothingRegionThickness =
            HybridSettings::getSmoothingRegionThickness();
        const auto pointChargeRadius = HybridSettings::getPointChargeRadius();

        if (com <= coreRadius)
            mol.setHybridZone(CORE);
        else if (com <= (layerRadius - smoothingRegionThickness))
            mol.setHybridZone(LAYER);
        else if (com <= layerRadius)
        {
            mol.setHybridZone(SMOOTHING);
            ++numberSmoothingMolecules;
        }
        else if (com <= layerRadius + pointChargeRadius)
            mol.setHybridZone(POINT_CHARGE);
        else
            mol.setHybridZone(OUTER);
    }

    setNumberSmoothingMolecules(numberSmoothingMolecules);
}

/**
 * @brief Deactivate atoms in molecules for inner region calculation
 *
 * This function iterates over all molecules in the simulation box and activates
 * or deactivates their atoms based on their hybrid zone and whether their index
 * is present in the inactiveMolecules set.
 *
 * - Atoms in molecules with hybrid zones CORE, LAYER, or SMOOTHING are
 * activated.
 * - Atoms in molecules whose index is in inactiveMolecules, or whose hybrid
 * zone is POINT_CHARGE or OUTER, are deactivated.
 *
 * @param inactiveMolecules Set of molecule indices to be deactivated regardless
 * of zone
 * @param simBox Simulation box containing the molecules
 */
void HybridConfigurator::deactivateMoleculesForInnerCalculation(
    std::unordered_set<int> inactiveMolecules,
    pq::SimBox&             simBox
)
{
    int count{0};

    for (auto& mol : simBox.getMolecules())
    {
        const auto hybridZone = mol.getHybridZone();

        if (hybridZone == CORE || hybridZone == LAYER ||
            hybridZone == SMOOTHING)
            mol.activateAtoms();
        else if (inactiveMolecules.contains(count) ||
                 hybridZone == POINT_CHARGE || hybridZone == OUTER)
            mol.deactivateAtoms();

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
 * @throw std::domain_error if a molecule is outside the smoothing region
 */
void HybridConfigurator::calculateSmoothingFactors(pq::SimBox& simBox)
{
    const auto layerRadius = HybridSettings::getLayerRadius();
    const auto smoothingRegionThickness =
        HybridSettings::getSmoothingRegionThickness();

    for (auto& mol : simBox.getSmoothingMolecules())
    {
        mol.calculateCenterOfMass(simBox.getBox());
        const auto com = norm(mol.getCenterOfMass());

        const double distanceFactor =
            (com - layerRadius + smoothingRegionThickness) /
            smoothingRegionThickness;

        if (distanceFactor < 0.0 || distanceFactor > 1.0)
            throw(std::domain_error(
                "Cannot calculate smoothing factor for molecule outside the "
                "smoothing region"
            ));

        const double dF = distanceFactor - 0.5;

        mol.setSmoothingFactor(
            dF * (dF * dF * (-6.0 * dF * dF + 0.5) - 1.875) + 0.5
        );
    }
}

/********************************
 * standard getters and setters *
 ********************************/

/**
 * @brief get the center of the inner region of the hybrid calculation
 *
 * @return Vec3D innerRegionCenter
 */
Vec3D HybridConfigurator::getInnerRegionCenter() { return _innerRegionCenter; }

/**
 * @brief get the number of molecules in the point charge region of the
 * hybrid calculation
 *
 * @return int numberPointChargeMolecules
 */
int HybridConfigurator::getNumberSmoothingMolecules()
{
    return _numberSmoothingMolecules;
}

/**
 * @brief set the center of the inner region of the hybrid calculation
 *
 * @param innerRegionCenter
 */
void HybridConfigurator::setInnerRegionCenter(Vec3D innerRegionCenter)
{
    _innerRegionCenter = innerRegionCenter;
}

/**
 * @brief set the number of molecules in the point charge region of the
 * hybrid calculation
 *
 * @param count
 */
void HybridConfigurator::setNumberSmoothingMolecules(int count)
{
    _numberSmoothingMolecules = count;
}
