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
     * @brief set the number of qm atoms
     *
     * @param nQMAtoms
     */
    void PhysicalData::setNumberOfQMAtoms(double nQMAtoms)
    {
        _numberOfQMAtoms = nQMAtoms;
    }

    /**
     * @brief set the loop time
     *
     * @param loopTime
     */
    void PhysicalData::setLoopTime(const Time& loopTime)
    {
        _loopTime = loopTime;
    }

    /**
     * @brief set the volume
     *
     * @param volume
     */
    void PhysicalData::setVolume(const Volume& volume) { _volume = volume; }

    /**
     * @brief set the density
     *
     * @param density
     */
    void PhysicalData::setDensity(const Density& density)
    {
        _density = density;
    }

    /**
     * @brief set the temperature
     *
     * @param temperature
     */
    void PhysicalData::setTemperature(const Temperature& temperature)
    {
        _temperature = temperature;
    }

    /**
     * @brief set the pressure
     *
     * @param pressure
     */
    void PhysicalData::setPressure(const Pressure& pressure)
    {
        _pressure = pressure;
    }

    /**
     * @brief set the coupled pressure
     *
     * @param coupledPressure
     */
    void PhysicalData::setCoupledPressure(const Pressure& coupledPressure)
    {
        _coupledPressure = coupledPressure;
    }

    /**
     * @brief set the virial
     *
     * @param virial
     */
    void PhysicalData::setVirial(const VirialPerMol& virial)
    {
        _virial = virial;
    }

    /**
     * @brief set the stress tensor
     *
     * @param stressTensor
     */
    void PhysicalData::setStressTensor(const Stress& stressTensor)
    {
        _stressTensor = stressTensor;
    }

    /**
     * @brief set the linear momentum
     *
     * @param momentum
     */
    void PhysicalData::setMomentum(const Momentum3D& momentum)
    {
        _momentum = momentum;
    }

    /**
     * @brief set the angular momentum
     *
     * @param angularMomentum
     */
    void PhysicalData::setAngularMomentum(
        const AngularMomentum3D& angularMomentum
    )
    {
        _angularMomentum = angularMomentum;
    }

    /**
     * @brief set the kinetic energy
     *
     * @param kineticEnergy
     */
    void PhysicalData::setKineticEnergy(const EnergyPerMol& kineticEnergy)
    {
        _kineticEnergy = kineticEnergy;
    }

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

    /**
     * @brief set the coulomb energy
     *
     * @param coulombEnergy
     */
    void PhysicalData::setCoulombEnergy(const EnergyPerMol& coulombEnergy)
    {
        _coulombEnergy = coulombEnergy;
    }

    /**
     * @brief set the non coulomb energy
     *
     * @param nonCoulombEnergy
     */
    void PhysicalData::setNonCoulombEnergy(const EnergyPerMol& nonCoulombEnergy)
    {
        _nonCoulombEnergy = nonCoulombEnergy;
    }

    /**
     * @brief set the intra coulomb energy
     *
     * @param intraCoulombEnergy
     */
    void PhysicalData::setIntraCoulombEnergy(
        const EnergyPerMol& intraCoulombEnergy
    )
    {
        _intraCoulombEnergy = intraCoulombEnergy;
    }

    /**
     * @brief set the intra non coulomb energy
     *
     * @param intraNonCoulombEnergy
     */
    void PhysicalData::setIntraNonCoulombEnergy(
        const EnergyPerMol& intraNonCoulombEnergy
    )
    {
        _intraNonCoulombEnergy = intraNonCoulombEnergy;
    }

    /**
     * @brief set the bond energy
     *
     * @param bondEnergy
     */
    void PhysicalData::setBondEnergy(const EnergyPerMol& bondEnergy)
    {
        _bondEnergy = bondEnergy;
    }

    /**
     * @brief set the angle energy
     *
     * @param angleEnergy
     */
    void PhysicalData::setAngleEnergy(const EnergyPerMol& angleEnergy)
    {
        _angleEnergy = angleEnergy;
    }

    /**
     * @brief set the dihedral energy
     *
     * @param dihedralEnergy
     */
    void PhysicalData::setDihedralEnergy(const EnergyPerMol& dihedralEnergy)
    {
        _dihedralEnergy = dihedralEnergy;
    }

    /**
     * @brief set the improper energy
     *
     * @param improperEnergy
     */
    void PhysicalData::setImproperEnergy(const EnergyPerMol& improperEnergy)
    {
        _improperEnergy = improperEnergy;
    }

    /**
     * @brief set the qm energy
     *
     * @param qmEnergy
     */
    void PhysicalData::setQMEnergy(const EnergyPerMol& qmEnergy)
    {
        _qmEnergy = qmEnergy;
    }

    /**
     * @brief set the qm reference energy
     *
     * @param qmReferenceEnergy
     */
    void PhysicalData::setQMReferenceEnergy(
        const EnergyPerMol& qmReferenceEnergy
    )
    {
        _qmReferenceEnergy = qmReferenceEnergy;
    }

    /**
     * @brief set the number of smoothing molecules
     *
     * @param numberSmMol
     */
    void PhysicalData::setNumberOfSmoothingMolecules(double numberSmMol)
    {
        _numberOfSmoothingMol = numberSmMol;
    }

    /**
     * @brief set nose hoover momentum energy
     *
     * @param momentumEnergy
     */
    void PhysicalData::setNoseHooverMomentumEnergy(
        const EnergyPerMol& momentumEnergy
    )
    {
        _noseHooverMomentumEnergy = momentumEnergy;
    }

    /**
     * @brief set nose hoover friction energy
     *
     * @param frictionEnergy
     */
    void PhysicalData::setNoseHooverFrictionEnergy(
        const Quantity& frictionEnergy
    )
    {
        _noseHooverFrictionEnergy = frictionEnergy;
    }

    /**
     * @brief set the lower distance constraints energy
     *
     * @param lowerDistanceConstraints
     */
    void PhysicalData::setLowerDistanceConstraints(
        const EnergyPerMol& lowerDistanceConstraints
    )
    {
        _lowerDistanceConstraints = lowerDistanceConstraints;
    }

    /**
     * @brief set the upper distance constraints energy
     *
     * @param upperDistanceConstraints
     */
    void PhysicalData::setUpperDistanceConstraints(
        const EnergyPerMol& upperDistanceConstraints
    )
    {
        _upperDistanceConstraints = upperDistanceConstraints;
    }

    /**
     * @brief set the ring polymer energy
     *
     * @param ringPolymerEnergy
     */
    void PhysicalData::setRingPolymerEnergy(
        const EnergyPerMol& ringPolymerEnergy
    )
    {
        _ringPolymerEnergy = ringPolymerEnergy;
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the number of qm atoms
     *
     * @return double
     */
    double PhysicalData::getNumberOfActiveAtoms() const
    {
        return _numberOfQMAtoms;
    }

    /**
     * @brief get the loop time
     *
     * @return const Time&
     */
    const Time& PhysicalData::getLoopTime() const { return _loopTime; }

    /**
     * @brief get the volume
     *
     * @return const Volume&
     */
    const Volume& PhysicalData::getVolume() const { return _volume; }

    /**
     * @brief get the density
     *
     * @return const Density&
     */
    const Density& PhysicalData::getDensity() const { return _density; }

    /**
     * @brief get the temperature
     *
     * @return const Temperature&
     */
    const Temperature& PhysicalData::getTemperature() const
    {
        return _temperature;
    }

    /**
     * @brief get the pressure
     *
     * @return const Pressure&
     */
    const Pressure& PhysicalData::getPressure() const { return _pressure; }

    /**
     * @brief get the coupled pressure
     *
     * @return const Pressure&
     */
    const Pressure& PhysicalData::getCoupledPressure() const
    {
        return _coupledPressure;
    }

    /**
     * @brief get the kinetic energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getKineticEnergy() const
    {
        return _kineticEnergy;
    }

    /**
     * @brief get the non coulomb energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getNonCoulombEnergy() const
    {
        return _nonCoulombEnergy;
    }

    /**
     * @brief get the coulomb energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getCoulombEnergy() const
    {
        return _coulombEnergy;
    }

    /**
     * @brief get the intra coulomb energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getIntraCoulombEnergy() const
    {
        return _intraCoulombEnergy;
    }

    /**
     * @brief get the intra non coulomb energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getIntraNonCoulombEnergy() const
    {
        return _intraNonCoulombEnergy;
    }

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
     * @brief get the bond energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getBondEnergy() const
    {
        return _bondEnergy;
    }

    /**
     * @brief get the angle energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getAngleEnergy() const
    {
        return _angleEnergy;
    }

    /**
     * @brief get the dihedral energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getDihedralEnergy() const
    {
        return _dihedralEnergy;
    }

    /**
     * @brief get the improper energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getImproperEnergy() const
    {
        return _improperEnergy;
    }

    /**
     * @brief get the qm energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getQMEnergy() const { return _qmEnergy; }

    /**
     * @brief get the qm reference energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getQMReferenceEnergy() const
    {
        return _qmReferenceEnergy;
    }

    /**
     * @brief get the number of smoothing molecules
     *
     * @return double
     */
    double PhysicalData::getNumberOfSmoothingMolecules() const
    {
        return _numberOfSmoothingMol;
    }

    /**
     * @brief get the nose hoover momentum energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getNoseHooverMomentumEnergy() const
    {
        return _noseHooverMomentumEnergy;
    }

    /**
     * @brief get the nose hoover friction energy
     *
     * @return const Quantity&
     */
    const Quantity& PhysicalData::getNoseHooverFrictionEnergy() const
    {
        return _noseHooverFrictionEnergy;
    }

    /**
     * @brief get the lower distance constraints energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getLowerDistanceConstraints() const
    {
        return _lowerDistanceConstraints;
    }

    /**
     * @brief get the upper distance constraints energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getUpperDistanceConstraints() const
    {
        return _upperDistanceConstraints;
    }

    /**
     * @brief get the ring polymer energy
     *
     * @return const EnergyPerMol&
     */
    const EnergyPerMol& PhysicalData::getRingPolymerEnergy() const
    {
        return _ringPolymerEnergy;
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

    /**
     * @brief get the virial
     *
     * @return const VirialPerMol&
     */
    const VirialPerMol& PhysicalData::getVirial() const { return _virial; }

    /**
     * @brief get the stress tensor
     *
     * @return const Stress&
     */
    const Stress& PhysicalData::getStressTensor() const
    {
        return _stressTensor;
    }

    /**
     * @brief get the linear momentum
     *
     * @return const Momentum3D&
     */
    const Momentum3D& PhysicalData::getMomentum() const { return _momentum; }

    /**
     * @brief get the angular momentum
     *
     * @return const AngularMomentum3D&
     */
    const AngularMomentum3D& PhysicalData::getAngularMomentum() const
    {
        return _angularMomentum;
    }

}   // namespace physicalData
