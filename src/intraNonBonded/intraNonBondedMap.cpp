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

#include <cstdlib>

#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"

namespace intraNonBonded
{

    /**
     * @brief Construct a new Intra Non Bonded Map:: Intra Non Bonded Map object
     *
     * @param molecule
     * @param intraNonBondedType
     */
    IntraNonBondedMap::IntraNonBondedMap(
        molsys::Molecule        *molecule,
        IntraNonBondedContainer *intraNonBondedType
    )
        : _molecule(molecule), _intraNonBondedContainer(intraNonBondedType)
    {
    }

    /**
     * @brief calculate the intra non bonded interactions for a single
     * intraNonBondedMap (for a single molecule)
     *
     * @param coulombPotential
     * @param nonCoulombPotential
     * @param simulationBox
     * @param physicalData
     */
    void IntraNonBondedMap::calculate(
        const pot::CoulombPotential *coulombPotential,
        pot::NonCoulombPotential    *nonCoulombPotential,
        const molsys::SimulationBox &simulationBox,
        physicalData::PhysicalData  &physicalData
    ) const
    {
        auto       coulombEnergy    = 0.0;
        auto       nonCoulombEnergy = 0.0;
        const auto box              = simulationBox.getBoxDimensions();

        const auto nAtomIndices =
            _intraNonBondedContainer->getAtomIndices().size();

        for (size_t atomIndex1 = 0; atomIndex1 < nAtomIndices; ++atomIndex1)
        {
            const auto atomIndices =
                _intraNonBondedContainer->getAtomIndices()[atomIndex1];

            for (const auto atomIndice : atomIndices)
            {
                const auto [coulombEnergyTemp, nonCoulombEnergyTemp] =
                    calculateSingleInteraction(
                        AtomIndex{atomIndex1},
                        atomIndice,
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
     * @brief calculate the intra non bonded interactions for a single atomic
     * pair within a single molecule
     *
     * @param atomIdx1
     * @param atomIdx2AsInt
     * @param box
     * @param coulPot
     * @param nonCoulPot
     * @return std::pair<double, double> - the coulomb and non-coulomb energy
     * for the interaction
     */
    std::pair<double, double> IntraNonBondedMap::calculateSingleInteraction(
        AtomIndex            atomIdx1,
        int                  atomIdx2AsInt,
        const linalg::Vec3D &box,
        physicalData::PhysicalData & /*physicalData*/,
        const pot::CoulombPotential *coulPot,
        pot::NonCoulombPotential    *nonCoulPot
    ) const
    {
        if (!_molecule->isActive())
            return {0.0, 0.0};

        auto coulombEnergy    = 0.0;
        auto nonCoulombEnergy = 0.0;

        const auto atomIdx2 =
            AtomIndex{static_cast<size_t>(::abs(atomIdx2AsInt))};
        const bool scale = atomIdx2AsInt < 0;

        const auto &pos1 = _molecule->getAtomPosition(atomIdx1);
        const auto &pos2 = _molecule->getAtomPosition(atomIdx2);

        auto       dPos = pos1 - pos2;
        const auto txyz = -box * round(dPos / box);
        // TODO: implement it more general via Box::calcShiftVector

        dPos                += txyz;
        const auto distance  = norm(dPos);

        if (distance < pot::CoulombPotential::getCoulombRadiusCutOff())
        {
            const auto charge1 = _molecule->getPartialCharge(atomIdx1);
            const auto charge2 = _molecule->getPartialCharge(atomIdx2);

            const auto chargeProduct = charge1 * charge2;

            auto [energy, force] = coulPot->calculate(distance, chargeProduct);

            if (scale)
            {
                const auto scaling =
                    settings::PotentialSettings::getScale14Coulomb();
                energy *= scaling;
                force  *= scaling;
            }
            coulombEnergy = energy;

            const auto atomType1 = _molecule->getAtomType(atomIdx1);
            const auto atomType2 = _molecule->getAtomType(atomIdx2);

            const auto globalVdwType1 =
                _molecule->getInternalGlobalVDWType(atomIdx1);
            const auto globalVdwType2 =
                _molecule->getInternalGlobalVDWType(atomIdx2);

            const auto moltype = _molecule->getMoltype();

            const std::tuple combinedIdx{
                moltype,
                moltype,
                atomType1,
                atomType2
            };

            const auto nonCoulombicPair = nonCoulPot->getNonCoulPair(
                combinedIdx,
                {globalVdwType1, globalVdwType2}
            );

            if (distance < nonCoulombicPair->getRadialCutOff())
            {
                auto [nonCoulombEnergyLocal, nonCoulombForce] =
                    nonCoulombicPair->calculate(distance);

                if (scale)
                {
                    const auto scaling =
                        settings::PotentialSettings::getScale14VDW();
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
     * @return molsys::Molecule*
     */
    molsys::Molecule *IntraNonBondedMap::getMolecule() const
    {
        return _molecule;
    }

    /**
     * @brief get the atom indices of the IntraNonBondedContainer object
     *
     * @return std::vector<std::vector<int>>
     */
    std::vector<std::vector<int>> IntraNonBondedMap::getAtomIndices() const
    {
        return _intraNonBondedContainer->getAtomIndices();
    }

}   // namespace intraNonBonded
