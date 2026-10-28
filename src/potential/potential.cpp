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

#include "box.hpp"
#include "coulombPotential.hpp"
#include "molecule.hpp"
#include "nonCoulombPotential.hpp"

namespace pot
{
    Potential::~Potential() = default;

    /**
     * @brief Calculate non-bonded inter-molecular interactions between two
     * atoms
     *
     * @details This template function computes both Coulombic and non-Coulombic
     * interactions between two atoms from different molecules. The charge types
     * are determined at compile time using template specialization, allowing
     * for efficient QM/MM hybrid calculations. The function applies periodic
     * boundary conditions, calculates distances, and evaluates both
     * electrostatic and van der Waals interactions within their respective
     * cutoff radii.
     *
     * @tparam ChargeTag1 Charge type for atom1 (QMChargeTag or MMChargeTag)
     * @tparam ChargeTag2 Charge type for atom2 (QMChargeTag or MMChargeTag)
     * @param box Simulation box for periodic boundary conditions
     * @param mol1 First molecule containing atom1
     * @param mol2 Second molecule containing atom2
     * @param atom1 First atom in the interaction pair
     * @param atom2 Second atom in the interaction pair
     * @return std::pair<double, double> Coulomb energy and non-Coulomb energy
     */
    template <typename ChargeTag1, typename ChargeTag2>
    std::pair<double, double> Potential::calculateSingleInteraction(
        const molsys::Box &box,
        molsys::Molecule  &mol1,
        molsys::Molecule  &mol2,
        molsys::Atom      &atom1,
        molsys::Atom      &atom2
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

            const std::tuple combinedIdx{
                moltype_i,
                moltype_j,
                atomType_i,
                atomType_j
            };

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto coulombPreFactor = charge_i * charge_j;

            auto [coulE, coulF] =
                _coulombPotential->calculate(distance, coulombPreFactor);
            coulombEnergy = coulE;

            const auto nonCoulPair = _nonCoulombPot->getNonCoulPair(
                combinedIdx,
                {globalVdwType_i, globalVdwType_j}
            );
            const auto rncCutOff = nonCoulPair->get_radialCutoff();

            if (distance < rncCutOff)
            {
                const auto &[nonCoulE, nonCoulF] =
                    nonCoulPair->calculate(distance);
                nonCoulombEnergy = nonCoulE;

                coulF += nonCoulF;
            }

            coulF /= distance;

            const auto forcexyz = coulF * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);
            atom2.addForce(-forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }

        return {coulombEnergy, nonCoulombEnergy};
    }

    template std::pair<double, double> pot::Potential::
        calculateSingleInteraction<pot::QMChargeTag, pot::MMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &
        ) const;

    template std::pair<double, double> pot::Potential::
        calculateSingleInteraction<pot::MMChargeTag, pot::QMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &
        ) const;

    template std::pair<double, double> pot::Potential::
        calculateSingleInteraction<pot::MMChargeTag, pot::MMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &
        ) const;

    /**
     * @brief Calculate single Coulomb interaction between two atoms
     *
     * @details This template function computes only the electrostatic
     * interaction between two atoms, ignoring van der Waals forces. It's
     * optimized for cases where only Coulombic interactions are needed. The
     * function applies periodic boundary conditions and respects the Coulomb
     * cutoff radius.
     *
     * @tparam ChargeTag1 Charge type for atom1 (QMChargeTag or MMChargeTag)
     * @tparam ChargeTag2 Charge type for atom2 (QMChargeTag or MMChargeTag)
     * @param box Simulation box for periodic boundary conditions
     * @param atom1 First atom in the interaction pair
     * @param atom2 Second atom in the interaction pair
     * @return double Coulomb energy of the interaction
     */
    template <typename ChargeTag1, typename ChargeTag2>
    double Potential::calculateSingleCoulombInteraction(
        const molsys::Box &box,
        molsys::Atom      &atom1,
        molsys::Atom      &atom2
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

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto coulombPreFactor = charge_i * charge_j;

            auto [coulE, coulF] =
                _coulombPotential->calculate(distance, coulombPreFactor);
            coulombEnergy = coulE;

            coulF /= distance;

            const auto forcexyz = coulF * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);
            atom2.addForce(-forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }

        return coulombEnergy;
    }

    template double pot::Potential::
        calculateSingleCoulombInteraction<pot::QMChargeTag, pot::MMChargeTag>(
            molsys::Box const &,
            molsys::Atom &,
            molsys::Atom &
        ) const;

    template double pot::Potential::
        calculateSingleCoulombInteraction<pot::MMChargeTag, pot::QMChargeTag>(
            molsys::Box const &,
            molsys::Atom &,
            molsys::Atom &
        ) const;

    /**
     * @brief Calculate one-way non-bonded interactions between two atoms
     *
     * @details This template function is similar to
     * calculateSingleInteraction but applies forces only to the first atom
     * (atom1). This is useful in specific algorithmic contexts where force
     * symmetry is handled elsewhere or when calculating interactions in a
     * one-directional manner. The function still computes both Coulombic
     * and non-Coulombic interactions but only updates forces and shift
     * forces for atom1.
     *
     * @tparam ChargeTag1 Charge type for atom1 (QMChargeTag or MMChargeTag)
     * @tparam ChargeTag2 Charge type for atom2 (QMChargeTag or MMChargeTag)
     * @param box Simulation box for periodic boundary conditions
     * @param mol1 First molecule containing atom1
     * @param mol2 Second molecule containing atom2
     * @param atom1 First atom in the interaction pair (receives forces)
     * @param atom2 Second atom in the interaction pair (no forces applied)
     * @return std::pair<double, double> Coulomb energy and non-Coulomb
     * energy
     */
    template <typename ChargeTag1, typename ChargeTag2>
    std::pair<double, double> Potential::calculateSingleInteractionOneWay(
        const molsys::Box &box,
        molsys::Molecule  &mol1,
        molsys::Molecule  &mol2,
        molsys::Atom      &atom1,
        molsys::Atom      &atom2
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

            const std::tuple combinedIdx{
                moltype_i,
                moltype_j,
                atomType_i,
                atomType_j
            };

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto coulombPreFactor = charge_i * charge_j;

            auto [coulE, coulF] =
                _coulombPotential->calculate(distance, coulombPreFactor);
            coulombEnergy = coulE;

            const auto nonCoulPair = _nonCoulombPot->getNonCoulPair(
                combinedIdx,
                {globalVdwType_i, globalVdwType_j}
            );
            const auto rncCutOff = nonCoulPair->get_radialCutoff();

            if (distance < rncCutOff)
            {
                const auto &[nonCoulE, nonCoulF] =
                    nonCoulPair->calculate(distance);
                nonCoulombEnergy = nonCoulE;

                coulF += nonCoulF;
            }

            coulF /= distance;

            const auto forcexyz = coulF * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }

        return {coulombEnergy, nonCoulombEnergy};
    }

    template std::pair<double, double> pot::Potential::
        calculateSingleInteractionOneWay<pot::MMChargeTag, pot::QMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &
        ) const;

    template std::pair<double, double> pot::Potential::
        calculateSingleInteractionOneWay<pot::QMChargeTag, pot::MMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &
        ) const;

    /**
     * @brief calculates QMMM forces by invoking core-to-outer, layer-to-outer,
     * and outer-to-outer force calculations
     *
     * @param simulationBox
     * @param physicalData
     * @param cellList
     * @param waterModelSettings
     */
    void Potential::calculateQMMMForces(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const molsys::CellList     &cellList,
        const WaterModelSettings   &waterModelSettings
    )
    {
        calculateCoreToOuterForces(
            simulationBox,
            physicalData,
            cellList,
            waterModelSettings
        );
        calculateLayerToOuterForces(
            simulationBox,
            physicalData,
            cellList,
            waterModelSettings
        );
        calculateOuterToOuterForces(
            simulationBox,
            physicalData,
            cellList,
            waterModelSettings
        );
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
        const std::shared_ptr<NonCoulombPotential> &pot
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
     * @return std::shared_ptr<pot::CoulombPotential>
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

}   // namespace pot
