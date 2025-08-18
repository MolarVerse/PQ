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

#include <stdexcept>   // for domain_error

#include "atom.hpp"             // for Atom
#include "hybridSettings.hpp"   // for HybridSettings
#include "simulationBox.hpp"    // for SimulationBox

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
    using enum HybridZone;
    auto numberPointChargeMolecules = getNumberPointChargeMolecules();

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
            mol.setHybridZone(SMOOTHING);
        else if (com <= layerRadius + pointChargeRadius)
        {
            mol.setHybridZone(POINT_CHARGE);
            ++numberPointChargeMolecules;
        }
        else
            mol.setHybridZone(OUTER);
    }

    setNumberPointChargeMolecules(numberPointChargeMolecules);
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
 * @brief get the number of molecules in the point charge region of the hybrid
 * calculation
 *
 * @return int numberPointChargeMolecules
 */
int HybridConfigurator::getNumberPointChargeMolecules()
{
    return _numberPointChargeMolecules;
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
 * @brief set the number of molecules in the point charge region of the hybrid
 * calculation
 *
 * @param count
 */
void HybridConfigurator::setNumberPointChargeMolecules(int count)
{
    _numberPointChargeMolecules = count;
}
