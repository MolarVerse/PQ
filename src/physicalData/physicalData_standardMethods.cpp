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

#include "physicalData.hpp"
#include "unit_types.hpp"

namespace physicalData
{
    /*************************
     *                       *
     * standard add methods  *
     *                       *
     *************************/

    /**
     * @brief add virial to the current virial stored in physical data
     *
     * @param virial
     */
    void PhysicalData::addVirial(const VirialPerMol& virial)
    {
        _virial += virial;
    }

    /**
     * @brief add QM energy to the current QM energy stored in physical
     * data
     *
     * @param qmEnergy
     */
    void PhysicalData::addQMEnergy(const EnergyPerMol& qmEnergy)
    {
        _qmEnergy += qmEnergy;
    }

    /**
     * @brief add QM reference energy to the current QM reference energy
     *
     * @param qmReferenceEnergy
     */
    void PhysicalData::addQMReferenceEnergy(
        const EnergyPerMol& qmReferenceEnergy
    )
    {
        _qmReferenceEnergy += qmReferenceEnergy;
    }

    /**
     * @brief add coulomb energy to the current coulomb energy stored in
     * physical data
     *
     * @param coulombEnergy
     */
    void PhysicalData::addCoulombEnergy(const EnergyPerMol& coulombEnergy)
    {
        _coulombEnergy += coulombEnergy;
    }

    /**
     * @brief add non coulomb energy to the current non coulomb energy stored in
     * physical data
     *
     * @param nonCoulombEnergy
     */
    void PhysicalData::addNonCoulombEnergy(const EnergyPerMol& nonCoulombEnergy)
    {
        _nonCoulombEnergy += nonCoulombEnergy;
    }

    /**
     * @brief add bond energy to the current bond energy stored in physical data
     *
     * @param bondEnergy
     */
    void PhysicalData::addBondEnergy(const EnergyPerMol& bondEnergy)
    {
        _bondEnergy += bondEnergy;
    }

    /**
     * @brief add angle energy to the current angle energy stored in physical
     * data
     *
     * @param angleEnergy
     */
    void PhysicalData::addAngleEnergy(const EnergyPerMol& angleEnergy)
    {
        _angleEnergy += angleEnergy;
    }

    /**
     * @brief add dihedral energy to the current dihedral energy stored in
     * physical data
     *
     * @param dihedralEnergy
     */
    void PhysicalData::addDihedralEnergy(const EnergyPerMol& dihedralEnergy)
    {
        _dihedralEnergy += dihedralEnergy;
    }

    /**
     * @brief add improper energy to the current improper energy stored in
     * physical data
     *
     * @param improperEnergy
     */
    void PhysicalData::addImproperEnergy(const EnergyPerMol& improperEnergy)
    {
        _improperEnergy += improperEnergy;
    }

    /**
     * @brief add ring polymer energy to the current ring polymer energy stored
     * in physical data
     *
     * @param ringPolymerEnergy
     */
    void PhysicalData::addRingPolymerEnergy(
        const EnergyPerMol& ringPolymerEnergy
    )
    {
        _ringPolymerEnergy += ringPolymerEnergy;
    }

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief set the kinetic energy atomic vector
     *
     * @param vec
     */
    void PhysicalData::setKineticEnergyAtomicVector(const VirialPerMol& vec)
    {
        _kinEnergyVirialTensor.atomic = vec;
    }

    /**
     * @brief set the kinetic energy molecular vector
     *
     * @param vec
     */
    void PhysicalData::setKineticEnergyMolecularVector(const VirialPerMol& vec)
    {
        _kinEnergyVirialTensor.molecular = vec;
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the intra energy
     *
     * @return EnergyPerMol
     */
    EnergyPerMol PhysicalData::getIntraEnergy() const
    {
        return _intraCoulombEnergy + _intraNonCoulombEnergy;
    }

    /**
     * @brief get the kinetic energy atomic vector
     *
     * @return linalg::tensor3D
     */
    const VirialPerMol& PhysicalData::getKinEnergyAtomTensor() const
    {
        return _kinEnergyVirialTensor.atomic;
    }

    /**
     * @brief get the kinetic energy molecular vector
     *
     * @return linalg::tensor3D
     */
    const VirialPerMol& PhysicalData::getKinEnergyMolTensor() const
    {
        return _kinEnergyVirialTensor.molecular;
    }

    /**
     * @brief get the kinetic energy virial tensor
     *
     * @param virialType - the virial type to get the kinetic energy virial
     * tensor for
     *
     * @return const VirialPerMol&
     */
    const VirialPerMol& PhysicalData::getKinEnergyVirialTensor(
        VirialType virialType
    ) const
    {
        return _kinEnergyVirialTensor.getVirialTensor(virialType);
    }

}   // namespace physicalData
