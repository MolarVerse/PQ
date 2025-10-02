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

#include "potential.hpp"

#include <cmath>   // for sqrt

#include "box.hpp"                   // for Box
#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential

using namespace potential;
using namespace simulationBox;
using namespace physicalData;

using enum ChargeType;

void Potential::calculateQMMMForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList      &cellList
)
{
    calculateForces(simBox, physicalData, cellList);
    calculateCoreToOuterForces(simBox, physicalData, cellList);
    calculateLayerToOuterForces(simBox, physicalData, cellList);
}

/**
 * @brief inner part of the double loop to calculate non-bonded inter molecular
 * interactions
 *
 * @param box
 * @param mol1
 * @param mol2
 * @param atom1
 * @param atom2
 * @param chargeType1 whether to use the QM charge or MM charge for atom1
 * @param chargeType2 whether to use the QM charge or MM charge for atom2
 * @return std::pair<double, double>
 */
std::pair<double, double> Potential::calculateSingleInteraction(
    const Box &box,
    Molecule  &mol1,
    Molecule  &mol2,
    Atom      &atom1,
    Atom      &atom2,
    ChargeType chargeType1,
    ChargeType chargeType2
) const
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto xyz_i = atom1.getPosition();
    const auto xyz_j = atom2.getPosition();

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -box.calcShiftVector(dxyz);

    dxyz += txyz;

    const double distanceSquared = normSquared(dxyz);

    if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff();
        distanceSquared < RcCutOff * RcCutOff)
    {
        const double distance   = ::sqrt(distanceSquared);
        const auto   atomType_i = atom1.getAtomType();
        const auto   atomType_j = atom2.getAtomType();

        const auto globalVdwType_i = atom1.getInternalGlobalVDWType();
        const auto globalVdwType_j = atom2.getInternalGlobalVDWType();

        const auto moltype_i = mol1.getMoltype();
        const auto moltype_j = mol2.getMoltype();

        const auto combinedIdx = {
            moltype_i,
            moltype_j,
            atomType_i,
            atomType_j,
            globalVdwType_i,
            globalVdwType_j
        };

        auto charge_i = 0.0;
        auto charge_j = 0.0;

        if (atom1.getQMCharge().has_value() && chargeType1 == QM_CHARGE)
            charge_i = atom1.getQMCharge().value();
        else
            charge_i = atom1.getPartialCharge();

        if (atom2.getQMCharge().has_value() && chargeType2 == QM_CHARGE)
            charge_j = atom2.getQMCharge().value();
        else
            charge_j = atom2.getPartialCharge();

        const auto coulombPreFactor = charge_i * charge_j;

        auto [e, f] = _coulombPotential->calculate(distance, coulombPreFactor);
        coulombEnergy = e;

        const auto nonCoulPair = _nonCoulombPot->getNonCoulPair(combinedIdx);
        const auto rncCutOff   = nonCoulPair->getRadialCutOff();

        if (distance < rncCutOff)
        {
            const auto &[nonCoulE, nonCoulF] = nonCoulPair->calculate(distance);
            nonCoulombEnergy                 = nonCoulE;

            f += nonCoulF;
        }

        f /= distance;

        const auto forcexyz = f * dxyz;

        const auto shiftForcexyz = forcexyz * txyz;

        atom1.addForce(forcexyz);
        atom2.addForce(-forcexyz);

        atom1.addShiftForce(shiftForcexyz);
    }

    return {coulombEnergy, nonCoulombEnergy};
}

/**
 * @brief calculate single Coulomb interaction between two atoms
 *
 * @param box simulation box for periodic boundary conditions
 * @param atom1 first atom
 * @param atom2 second atom
 * @param chargeType1 whether to use the QM charge or MM charge for atom1
 * @param chargeType2 whether to use the QM charge or MM charge for atom2
 * @return double Coulomb energy
 */
double Potential::calculateSingleCoulombInteraction(
    const Box &box,
    Atom      &atom1,
    Atom      &atom2,
    ChargeType chargeType1,
    ChargeType chargeType2
) const
{
    auto coulombEnergy = 0.0;

    const auto xyz_i = atom1.getPosition();
    const auto xyz_j = atom2.getPosition();

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -box.calcShiftVector(dxyz);

    dxyz += txyz;

    const double distanceSquared = normSquared(dxyz);

    if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff();
        distanceSquared < RcCutOff * RcCutOff)
    {
        const double distance = ::sqrt(distanceSquared);

        auto charge_i = 0.0;
        auto charge_j = 0.0;

        if (atom1.getQMCharge().has_value() && chargeType1 == QM_CHARGE)
            charge_i = atom1.getQMCharge().value();
        else
            charge_i = atom1.getPartialCharge();

        if (atom2.getQMCharge().has_value() && chargeType2 == QM_CHARGE)
            charge_j = atom2.getQMCharge().value();
        else
            charge_j = atom2.getPartialCharge();

        const auto coulombPreFactor = charge_i * charge_j;

        auto [e, f] = _coulombPotential->calculate(distance, coulombPreFactor);
        coulombEnergy = e;

        f /= distance;

        const auto forcexyz = f * dxyz;

        const auto shiftForcexyz = forcexyz * txyz;

        atom1.addForce(forcexyz);
        atom2.addForce(-forcexyz);

        atom1.addShiftForce(shiftForcexyz);
    }

    return coulombEnergy;
}

std::pair<double, double> Potential::calculateSingleInteractionOneWay(
    const Box &box,
    Molecule  &mol1,
    Molecule  &mol2,
    Atom      &atom1,
    Atom      &atom2,
    ChargeType chargeType1,
    ChargeType chargeType2
) const
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto xyz_i = atom1.getPosition();
    const auto xyz_j = atom2.getPosition();

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -box.calcShiftVector(dxyz);

    dxyz += txyz;

    const double distanceSquared = normSquared(dxyz);

    if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff();
        distanceSquared < RcCutOff * RcCutOff)
    {
        const double distance   = ::sqrt(distanceSquared);
        const auto   atomType_i = atom1.getAtomType();
        const auto   atomType_j = atom2.getAtomType();

        const auto globalVdwType_i = atom1.getInternalGlobalVDWType();
        const auto globalVdwType_j = atom2.getInternalGlobalVDWType();

        const auto moltype_i = mol1.getMoltype();
        const auto moltype_j = mol2.getMoltype();

        const auto combinedIdx = {
            moltype_i,
            moltype_j,
            atomType_i,
            atomType_j,
            globalVdwType_i,
            globalVdwType_j
        };

        auto charge_i = 0.0;
        auto charge_j = 0.0;

        if (atom1.getQMCharge().has_value() && chargeType1 == QM_CHARGE)
            charge_i = atom1.getQMCharge().value();
        else
            charge_i = atom1.getPartialCharge();

        if (atom2.getQMCharge().has_value() && chargeType2 == QM_CHARGE)
            charge_j = atom2.getQMCharge().value();
        else
            charge_j = atom2.getPartialCharge();

        const auto coulombPreFactor = charge_i * charge_j;

        auto [e, f] = _coulombPotential->calculate(distance, coulombPreFactor);
        coulombEnergy = e;

        const auto nonCoulPair = _nonCoulombPot->getNonCoulPair(combinedIdx);
        const auto rncCutOff   = nonCoulPair->getRadialCutOff();

        if (distance < rncCutOff)
        {
            const auto &[nonCoulE, nonCoulF] = nonCoulPair->calculate(distance);
            nonCoulombEnergy                 = nonCoulE;

            f += nonCoulF;
        }

        f /= distance;

        const auto forcexyz = f * dxyz;

        const auto shiftForcexyz = forcexyz * txyz;

        atom1.addForce(forcexyz);

        atom1.addShiftForce(shiftForcexyz);
    }

    return {coulombEnergy, nonCoulombEnergy};
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the coulomb potential as a shared pointer
 *
 * @param pot
 */
void Potential::setNonCoulombPotential(
    const std::shared_ptr<NonCoulombPotential> pot
)
{
    _nonCoulombPot = pot;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief get the coulomb potential
 *
 * @return CoulombPotential&
 */
CoulombPotential &Potential::getCoulombPotential() const
{
    return *_coulombPotential;
}

/**
 * @brief get the non-coulomb potential
 *
 * @return NonCoulombPotential&
 */
NonCoulombPotential &Potential::getNonCoulombPotential() const
{
    return *_nonCoulombPot;
}

/**
 * @brief get the coulomb potential as a shared pointer
 *
 * @return SharedCoulombPot
 */
std::shared_ptr<CoulombPotential> Potential::getCoulombPotSharedPtr() const
{
    return _coulombPotential;
}

/**
 * @brief get the non-coulomb potential as a shared pointer
 *
 * @return SharedNonCoulombPot
 */
std::shared_ptr<NonCoulombPotential> Potential::getNonCoulombPotSharedPtr(
) const
{
    return _nonCoulombPot;
}