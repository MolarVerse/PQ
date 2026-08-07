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

#include "intraNonBondedMap.hpp"

#include <cstdlib>   // for abs, size_t
#include <memory>    // for __shared_ptr_access, shared_ptr

#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "physicalData.hpp"          // for PhysicalData
#include "potentialSettings.hpp"     // for PotentialSettings
#include "simulationBox.hpp"         // for SimulationBox
#include "vector3d.hpp"              // for norm, operator*, Vector3D

using namespace intraNonBonded;
using namespace potential;
using namespace physicalData;
using namespace simulationBox;
using namespace linearAlgebra;
using namespace settings;

/**
 * @brief Construct a new Intra Non Bonded Map:: Intra Non Bonded Map object
 *
 * @param molecule
 * @param intraNonBondedType
 */
IntraNonBondedMap::IntraNonBondedMap(
    pq::Molecule            *molecule,
    IntraNonBondedContainer *intraNonBondedType
)
    : _molecule(molecule), _intraNonBondedContainer(intraNonBondedType){};

/**
 * @brief calculate the intra non bonded interactions for a single
 * intraNonBondedMap (for a single molecule)
 *
 * @param coulombPotential
 * @param nonCoulombPotential
 * @param box
 * @param physicalData
 */
void IntraNonBondedMap::calculate(
    const CoulombPotential *coulombPotential,
    NonCoulombPotential    *nonCoulombPotential,
    const SimulationBox    &simulationBox,
    PhysicalData           &physicalData
) const
{
    auto       coulombEnergy    = 0.0;
    auto       nonCoulombEnergy = 0.0;
    const auto box              = simulationBox.getBoxDimensions();

    const auto nAtomIndices = _intraNonBondedContainer->getAtomIndices().size();

    for (size_t atomIndex1 = 0; atomIndex1 < nAtomIndices; ++atomIndex1)
    {
        const auto atomIndices =
            _intraNonBondedContainer->getAtomIndices()[atomIndex1];

        for (auto iter = atomIndices.begin(); iter != atomIndices.end(); ++iter)
        {
            const auto [coulombEnergyTemp, nonCoulombEnergyTemp] =
                calculateSingleInteraction(
                    atomIndex1,
                    *iter,
                    box,
                    physicalData,
                    coulombPotential,
                    nonCoulombPotential
                );

            coulombEnergy    += coulombEnergyTemp;
            nonCoulombEnergy += nonCoulombEnergyTemp;
        }
    }

    physicalData.addIntraCoulombEnergy(coulombEnergy);
    physicalData.addIntraNonCoulombEnergy(nonCoulombEnergy);
}

/**
 * @brief calculate the intra non bonded interactions for a single atomic pair
 * within a single molecule
 *
 * @param atomIndex1
 * @param atomIndex2AsInt
 * @param box
 * @param coulombPotential
 * @param nonCoulombPotential
 * @return std::pair<double, double> - the coulomb and non-coulomb energy for
 * the interaction
 */
std::pair<double, double> IntraNonBondedMap::calculateSingleInteraction(
    const size_t atomIdx1,
    const int    atomIndex2AsInt,
    const Vec3D &box,
    PhysicalData &,
    const CoulombPotential *coulPot,
    NonCoulombPotential    *nonCoulPot
) const
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto atomIdx2 = size_t(::abs(atomIndex2AsInt));
    const bool scale    = atomIndex2AsInt < 0;

    const auto &pos1 = _molecule->getAtomPosition(atomIdx1);
    const auto &pos2 = _molecule->getAtomPosition(atomIdx2);

    auto       dPos = pos1 - pos2;
    const auto txyz = -box * round(dPos / box);
    // TODO: implement it more general via Box::calcShiftVector

    dPos                += txyz;
    const auto distance  = norm(dPos);

    if (distance < CoulombPotential::getCoulombRadiusCutOff())
    {
        const auto charge1 = _molecule->getPartialCharge(size_t(atomIdx1));
        const auto charge2 = _molecule->getPartialCharge(atomIdx2);

        const auto chargeProduct = charge1 * charge2;

        auto [energy, force] = coulPot->calculate(distance, chargeProduct);

        if (scale)
        {
            const auto scaling  = PotentialSettings::getScale14Coulomb();
            energy             *= scaling;
            force              *= scaling;
        }
        coulombEnergy = energy;

        const size_t atomType1 = _molecule->getAtomType(atomIdx1);
        const size_t atomType2 = _molecule->getAtomType(atomIdx2);

        // clang-format off
        const auto globalVdwType1 = _molecule->getInternalGlobalVDWType(atomIdx1);
        const auto globalVdwType2 = _molecule->getInternalGlobalVDWType(atomIdx2);
        // clang-format on

        const auto moltype = _molecule->getMoltype();

        const auto combinedIdx = {
            moltype,
            moltype,
            atomType1,
            atomType2,
            globalVdwType1,
            globalVdwType2
        };

        const auto nonCoulombicPair = nonCoulPot->getNonCoulPair(combinedIdx);

        if (distance < nonCoulombicPair->getRadialCutOff())
        {
            auto [nonCoulombEnergyLocal, nonCoulombForce] =
                nonCoulombicPair->calculate(distance);

            if (scale)
            {
                const auto scaling     = PotentialSettings::getScale14VDW();
                nonCoulombEnergyLocal *= scaling;
                nonCoulombForce       *= scaling;
            }

            nonCoulombEnergy  = nonCoulombEnergyLocal;
            force            += nonCoulombForce;
        }

        force /= distance;

        const auto forcexyz = force * dPos;

        const auto shiftForcexyz = forcexyz * txyz;

        _molecule->addAtomForce(atomIdx1, forcexyz);
        _molecule->addAtomForce(atomIdx2, -forcexyz);

        _molecule->addAtomShiftForce(atomIdx1, shiftForcexyz);
    }

    return {coulombEnergy, nonCoulombEnergy};
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the IntraNonBondedContainer object
 *
 * @return IntraNonBondedContainer*
 */
IntraNonBondedContainer *IntraNonBondedMap::getIntraNonBondedType() const
{
    return _intraNonBondedContainer;
}

/**
 * @brief get the molecule pointer
 *
 * @return pq::Molecule*
 */
pq::Molecule *IntraNonBondedMap::getMolecule() const { return _molecule; }

/**
 * @brief get the atom indices of the IntraNonBondedContainer object
 *
 * @return std::vector<std::vector<int>>
 */
std::vector<std::vector<int>> IntraNonBondedMap::getAtomIndices() const
{
    return _intraNonBondedContainer->getAtomIndices();
}