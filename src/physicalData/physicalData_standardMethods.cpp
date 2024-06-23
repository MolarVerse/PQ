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

using namespace physicalData;

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
void PhysicalData::addVirial(const linearAlgebra::tensor3D &virial)
{
    _virial += virial;
}

/**
 * @brief add coulomb energy to the current coulomb energy stored in physical
 * data
 *
 * @param coulombEnergy
 */
void PhysicalData::addCoulombEnergy(const double coulombEnergy)
{
    _coulombEnergy += coulombEnergy;
}

/**
 * @brief add non coulomb energy to the current non coulomb energy stored in
 * physical data
 *
 * @param nonCoulombEnergy
 */
void PhysicalData::addNonCoulombEnergy(const double nonCoulombEnergy)
{
    _nonCoulombEnergy += nonCoulombEnergy;
}

/**
 * @brief add bond energy to the current bond energy stored in physical data
 *
 * @param bondEnergy
 */
void PhysicalData::addBondEnergy(const double bondEnergy)
{
    _bondEnergy += bondEnergy;
}

/**
 * @brief add angle energy to the current angle energy stored in physical data
 *
 * @param angleEnergy
 */
void PhysicalData::addAngleEnergy(const double angleEnergy)
{
    _angleEnergy += angleEnergy;
}

/**
 * @brief add dihedral energy to the current dihedral energy stored in physical
 * data
 *
 * @param dihedralEnergy
 */
void PhysicalData::addDihedralEnergy(const double dihedralEnergy)
{
    _dihedralEnergy += dihedralEnergy;
}

/**
 * @brief add improper energy to the current improper energy stored in physical
 * data
 *
 * @param improperEnergy
 */
void PhysicalData::addImproperEnergy(const double improperEnergy)
{
    _improperEnergy += improperEnergy;
}

/**
 * @brief add ring polymer energy to the current ring polymer energy stored in
 * physical data
 *
 * @param ringPolymerEnergy
 */
void PhysicalData::addRingPolymerEnergy(const double ringPolymerEnergy)
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
void PhysicalData::setNumberOfQMAtoms(const double nQMAtoms)
{
    _numberOfQMAtoms = nQMAtoms;
}

/**
 * @brief set the loop time
 *
 * @param loopTime
 */
void PhysicalData::setLoopTime(const double loopTime) { _loopTime = loopTime; }

/**
 * @brief set the volume
 *
 * @param volume
 */
void PhysicalData::setVolume(const double volume) { _volume = volume; }

/**
 * @brief set the density
 *
 * @param density
 */
void PhysicalData::setDensity(const double density) { _density = density; }

/**
 * @brief set the temperature
 *
 * @param temperature
 */
void PhysicalData::setTemperature(const double temperature)
{
    _temperature = temperature;
}

/**
 * @brief set the pressure
 *
 * @param pressure
 */
void PhysicalData::setPressure(const double pressure) { _pressure = pressure; }

/**
 * @brief set the virial
 *
 * @param virial
 */
void PhysicalData::setVirial(const linearAlgebra::tensor3D &virial)
{
    _virial = virial;
}

/**
 * @brief set the stress tensor
 *
 * @param stressTensor
 */
void PhysicalData::setStressTensor(const linearAlgebra::tensor3D &stressTensor)
{
    _stressTensor = stressTensor;
}

/**
 * @brief set the linear momentum
 *
 * @param momentum
 */
void PhysicalData::setMomentum(const linearAlgebra::Vec3D &momentum)
{
    _momentum = momentum;
}

/**
 * @brief set the angular momentum
 *
 * @param vec
 */
void PhysicalData::setAngularMomentum(
    const linearAlgebra::Vec3D &angularMomentum
)
{
    _angularMomentum = angularMomentum;
}

/**
 * @brief set the kinetic energy
 *
 * @param kineticEnergy
 */
void PhysicalData::setKineticEnergy(const double kineticEnergy)
{
    _kineticEnergy = kineticEnergy;
}

/**
 * @brief set the kinetic energy atomic vector
 *
 * @param vec
 */
void PhysicalData::setKineticEnergyAtomicVector(
    const linearAlgebra::tensor3D &vec
)
{
    _kineticEnergyAtomicTensor = vec;
}

/**
 * @brief set the kinetic energy molecular vector
 *
 * @param vec
 */
void PhysicalData::setKineticEnergyMolecularVector(
    const linearAlgebra::tensor3D &vec
)
{
    _kineticEnergyMolecularTensor = vec;
}

/**
 * @brief set the coulomb energy
 *
 * @param coulombEnergy
 */
void PhysicalData::setCoulombEnergy(const double coulombEnergy)
{
    _coulombEnergy = coulombEnergy;
}

/**
 * @brief set the non coulomb energy
 *
 * @param nonCoulombEnergy
 */
void PhysicalData::setNonCoulombEnergy(const double nonCoulombEnergy)
{
    _nonCoulombEnergy = nonCoulombEnergy;
}

/**
 * @brief set the intra coulomb energy
 *
 * @param intraCoulombEnergy
 */
void PhysicalData::setIntraCoulombEnergy(const double intraCoulombEnergy)
{
    _intraCoulombEnergy = intraCoulombEnergy;
}

/**
 * @brief set the intra non coulomb energy
 *
 * @param intraNonCoulombEnergy
 */
void PhysicalData::setIntraNonCoulombEnergy(const double intraNonCoulombEnergy)
{
    _intraNonCoulombEnergy = intraNonCoulombEnergy;
}

/**
 * @brief set the bond energy
 *
 * @param bondEnergy
 */
void PhysicalData::setBondEnergy(const double bondEnergy)
{
    _bondEnergy = bondEnergy;
}

/**
 * @brief set the angle energy
 *
 * @param angleEnergy
 */
void PhysicalData::setAngleEnergy(const double angleEnergy)
{
    _angleEnergy = angleEnergy;
}

/**
 * @brief set the dihedral energy
 *
 * @param dihedralEnergy
 */
void PhysicalData::setDihedralEnergy(const double dihedralEnergy)
{
    _dihedralEnergy = dihedralEnergy;
}

/**
 * @brief set the improper energy
 *
 * @param improperEnergy
 */
void PhysicalData::setImproperEnergy(const double improperEnergy)
{
    _improperEnergy = improperEnergy;
}

/**
 * @brief set the qm energy
 *
 * @param qmEnergy
 */
void PhysicalData::setQMEnergy(const double qmEnergy) { _qmEnergy = qmEnergy; }

/**
 * @brief set nose hoover momentum energy
 *
 * @param momentumEnergy
 */
void PhysicalData::setNoseHooverMomentumEnergy(const double momentumEnergy)
{
    _noseHooverMomentumEnergy = momentumEnergy;
}

/**
 * @brief set nose hoover friction energy
 *
 * @param frictionEnergy
 */
void PhysicalData::setNoseHooverFrictionEnergy(const double frictionEnergy)
{
    _noseHooverFrictionEnergy = frictionEnergy;
}

/**
 * @brief set the lower distance constraints energy
 *
 * @param lowerDistanceConstraints
 */
void PhysicalData::setLowerDistanceConstraints(
    const double lowerDistanceConstraints
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
    const double upperDistanceConstraints
)
{
    _upperDistanceConstraints = upperDistanceConstraints;
}

/**
 * @brief set the ring polymer energy
 *
 * @param ringPolymerEnergy
 */
void PhysicalData::setRingPolymerEnergy(const double ringPolymerEnergy)
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
double PhysicalData::getNumberOfQMAtoms() const { return _numberOfQMAtoms; }

/**
 * @brief get the loop time
 *
 * @return double
 */
double PhysicalData::getLoopTime() const { return _loopTime; }

/**
 * @brief get the volume
 *
 * @return double
 */
double PhysicalData::getVolume() const { return _volume; }

/**
 * @brief get the density
 *
 * @return double
 */
double PhysicalData::getDensity() const { return _density; }

/**
 * @brief get the temperature
 *
 * @return double
 */
double PhysicalData::getTemperature() const { return _temperature; }

/**
 * @brief get the pressure
 *
 * @return double
 */
double PhysicalData::getPressure() const { return _pressure; }

/**
 * @brief get the kinetic energy
 *
 * @return double
 */
double PhysicalData::getKineticEnergy() const { return _kineticEnergy; }

/**
 * @brief get the non coulomb energy
 *
 * @return double
 */
double PhysicalData::getNonCoulombEnergy() const { return _nonCoulombEnergy; }

/**
 * @brief get the coulomb energy
 *
 * @return double
 */
double PhysicalData::getCoulombEnergy() const { return _coulombEnergy; }

/**
 * @brief get the intra coulomb energy
 *
 * @return double
 */
double PhysicalData::getIntraCoulombEnergy() const
{
    return _intraCoulombEnergy;
}

/**
 * @brief get the intra non coulomb energy
 *
 * @return double
 */
double PhysicalData::getIntraNonCoulombEnergy() const
{
    return _intraNonCoulombEnergy;
}

/**
 * @brief get the intra energy
 *
 * @return double
 */
double PhysicalData::getIntraEnergy() const
{
    return _intraCoulombEnergy + _intraNonCoulombEnergy;
}

/**
 * @brief get the bond energy
 *
 * @return double
 */
double PhysicalData::getBondEnergy() const { return _bondEnergy; }

/**
 * @brief get the angle energy
 *
 * @return double
 */
double PhysicalData::getAngleEnergy() const { return _angleEnergy; }

/**
 * @brief get the dihedral energy
 *
 * @return double
 */
double PhysicalData::getDihedralEnergy() const { return _dihedralEnergy; }

/**
 * @brief get the improper energy
 *
 * @return double
 */
double PhysicalData::getImproperEnergy() const { return _improperEnergy; }

/**
 * @brief get the qm energy
 *
 * @return double
 */
double PhysicalData::getQMEnergy() const { return _qmEnergy; }

/**
 * @brief get the nose hoover momentum energy
 *
 * @return double
 */
double PhysicalData::getNoseHooverMomentumEnergy() const
{
    return _noseHooverMomentumEnergy;
}

/**
 * @brief get the nose hoover friction energy
 *
 * @return double
 */
double PhysicalData::getNoseHooverFrictionEnergy() const
{
    return _noseHooverFrictionEnergy;
}

/**
 * @brief get the lower distance constraints energy
 *
 * @return double
 */
double PhysicalData::getLowerDistanceConstraints() const
{
    return _lowerDistanceConstraints;
}

/**
 * @brief get the upper distance constraints energy
 *
 * @return double
 */
double PhysicalData::getUpperDistanceConstraints() const
{
    return _upperDistanceConstraints;
}

/**
 * @brief get the ring polymer energy
 *
 * @return double
 */
double PhysicalData::getRingPolymerEnergy() const { return _ringPolymerEnergy; }

/**
 * @brief get the kinetic energy atomic vector
 *
 * @return linearAlgebra::tensor3D
 */
linearAlgebra::tensor3D PhysicalData::getKineticEnergyAtomicVector() const
{
    return _kineticEnergyAtomicTensor;
}

/**
 * @brief get the kinetic energy molecular vector
 *
 * @return linearAlgebra::tensor3D
 */
linearAlgebra::tensor3D PhysicalData::getKineticEnergyMolecularVector() const
{
    return _kineticEnergyMolecularTensor;
}

/**
 * @brief get the virial
 *
 * @return linearAlgebra::tensor3D
 */
linearAlgebra::tensor3D PhysicalData::getVirial() const { return _virial; }

/**
 * @brief get the stress tensor
 *
 * @return linearAlgebra::tensor3D
 */
linearAlgebra::tensor3D PhysicalData::getStressTensor() const
{
    return _stressTensor;
}

/**
 * @brief get the linear momentum
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D PhysicalData::getMomentum() const { return _momentum; }

/**
 * @brief get the angular momentum
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D PhysicalData::getAngularMomentum() const
{
    return _angularMomentum;
}