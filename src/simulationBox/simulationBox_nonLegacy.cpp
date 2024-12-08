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

#include "constants.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

using namespace simulationBox;
using namespace constants;
using namespace settings;
using namespace linearAlgebra;

/**
 * @brief calculate total mass of simulationBox
 *
 */
void SimulationBox::calculateTotalMass()
{
    _totalMass = 0.0;

    const auto *const massPtr = getMassesPtr();

    // clang-format off

    #pragma omp target teams distribute parallel for \
                is_device_ptr(massPtr)               \
                map(_totalMass)                      \
                reduction(+:_totalMass)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        _totalMass += massPtr[i];

    // clang-format on
}

/**
 * @brief calculate total force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateTotalForce()
{
    auto fx = 0.0;
    auto fy = 0.0;
    auto fz = 0.0;

    flattenForces();

    const auto *const forcesPtr = getForcesPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for \
                is_device_ptr(forcesPtr)              \
                map(fx, fy, fz)                       \
                reduction(+:fx, fy, fz)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
    {
        fx += forcesPtr[i * 3];
        fy += forcesPtr[i * 3 + 1];
        fz += forcesPtr[i * 3 + 2];
    }
    // clang-format on

    const auto totalForce = Vec3D{fx, fy, fz};

    return norm(totalForce);
}

/**
 * @brief get the positions ptr
 *
 * @details This method is used to get a pointer to the positions either on the
 * CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the positions on the GPU is returned. Otherwise, the pointer to
 * the positions on the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getPosPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _posDevice;
    else
#endif
        return _pos.data();
}

/**
 * @brief get the velocities ptr
 *
 * @details This method is used to get a pointer to the velocities either on the
 * CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the velocities on the GPU is returned. Otherwise, the pointer to
 * the velocities on the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getVelPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _velDevice;
    else
#endif
        return _vel.data();
}

/**
 * @brief get the forces ptr
 *
 * @details This method is used to get a pointer to the forces either on the CPU
 * or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the pointer
 * to the forces on the GPU is returned. Otherwise, the pointer to the forces on
 * the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getForcesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _forcesDevice;
    else
#endif
        return _forces.data();
}

/**
 * @brief get the shift forces ptr
 *
 * @details This method is used to get a pointer to the shift forces either on
 * the CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the shift forces on the GPU is returned. Otherwise, the pointer to
 * the shift forces on the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getShiftForcesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _shiftForcesDevice;
    else
#endif
        return _shiftForces.data();
}

/**
 * @brief get the old positions ptr
 *
 * @details This method is used to get a pointer to the old positions either on
 * the CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the old positions on the GPU is returned. Otherwise, the pointer
 * to the old positions on the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getOldPosPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _oldPosDevice;
    else
#endif
        return _oldPos.data();
}

/**
 * @brief get the old velocities ptr
 *
 * @details This method is used to get a pointer to the old velocities either on
 * the CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the old velocities on the GPU is returned. Otherwise, the pointer
 * to the old velocities on the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getOldVelPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _oldVelDevice;
    else
#endif
        return _oldVel.data();
}

/**
 * @brief get the old forces ptr
 *
 * @details This method is used to get a pointer to the old forces either on the
 * CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the old forces on the GPU is returned. Otherwise, the pointer to
 * the old forces on the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getOldForcesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _oldForcesDevice;
    else
#endif
        return _oldForces.data();
}

/**
 * @brief get the masses ptr
 *
 * @details This method is used to get a pointer to the masses either on the
 * CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the masses on the GPU is returned. Otherwise, the pointer to the
 * masses on the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getMassesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _massesDevice;
    else
#endif
        return _masses.data();
}

/**
 * @brief get the charges ptr
 *
 * @details This method is used to get a pointer to the charges either on the
 * CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the charges on the GPU is returned. Otherwise, the pointer to the
 * charges on the CPU is returned.
 *
 * @return const Real*
 */
Real *SimulationBox::getChargesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _chargesDevice;
    else
#endif
        return _charges.data();
}

/**
 * @brief get the atom types ptr
 *
 * @details This method is used to get a pointer to the atom types either on the
 * CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the atom types on the GPU is returned. Otherwise, the pointer to
 * the atom types on the CPU is returned.
 *
 * @return const size_t*
 */
size_t *SimulationBox::getAtomTypesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _atomTypesDevice;
    else
#endif
        return _atomTypes.data();
}

/**
 * @brief get the molecule types ptr
 *
 * @details This method is used to get a pointer to the molecule types either on
 * the CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag, the
 * pointer to the molecule types on the GPU is returned. Otherwise, the pointer
 * to the molecule types on the CPU is returned.
 *
 * @return const size_t*
 */
size_t *SimulationBox::getMolTypesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _molTypesDevice;
    else
#endif
        return _molTypes.data();
}

/**
 * @brief get the atoms per molecule ptr
 *
 * @details This method is used to get a pointer to the atoms per molecule
 * either on the CPU or on the GPU. If the code is compiled with the __PQ_GPU__
 * flag, the pointer to the atoms per molecule on the GPU is returned.
 * Otherwise, the pointer to the atoms per molecule on the CPU is returned.
 *
 * @return const size_t*
 */
size_t *SimulationBox::getAtomsPerMoleculePtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _atomsPerMoleculeDevice;
    else
#endif
        return _atomsPerMolecule.data();
}

/**
 * @brief get the molecule indices ptr
 *
 * @details This method is used to get a pointer to the molecule indices either
 * on the CPU or on the GPU. If the code is compiled with the __PQ_GPU__ flag,
 * the pointer to the molecule indices on the GPU is returned. Otherwise, the
 * pointer to the molecule indices on the CPU is returned.
 *
 * @return const size_t*
 */
size_t *SimulationBox::getMoleculeIndicesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _moleculeIndicesDevice;
    else
#endif
        return _moleculeIndices.data();
}

/**
 * @brief get the internal global VDW types ptr
 *
 * @details This method is used to get a pointer to the internal global VDW
 * types either on the CPU or on the GPU. If the code is compiled with the
 * __PQ_GPU__ flag, the pointer to the internal global VDW types on the GPU is
 * returned. Otherwise, the pointer to the internal global VDW types on the CPU
 * is returned.
 *
 * @return const size_t*
 */
size_t *SimulationBox::getInternalGlobalVDWTypesPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _internalGlobalVDWTypesDevice;
    else
#endif
        return _internalGlobalVDWTypes.data();
}

/**
 * @brief flattens positions of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 *
 * @TODO: this method should not return anything, but rather only set the _pos
 * member variable
 */
std::vector<Real> SimulationBox::flattenPositions()
{
    if (_pos.size() != _atoms.size() * 3)
        _pos.resize(_atoms.size() * 3);

    Real *const positions = _pos.data();

    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            positions[i * 3 + j] = atom->getPosition()[j];
    }
    // clang-format on

    return _pos;
}

/**
 * @brief flattens velocities of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 *
 * @TODO: this method should not return anything, but rather only set the _vel
 */
std::vector<Real> SimulationBox::flattenVelocities()
{
    if (_vel.size() != _atoms.size() * 3)
        _vel.resize(_atoms.size() * 3);

    Real *const velocities = _vel.data();

    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            velocities[i * 3 + j] = atom->getVelocity()[j];
    }
    // clang-format on

    return _vel;
}

/**
 * @brief flattens forces of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 *
 * @TODO: this method should not return anything, but rather only set the
 * _forces
 */
std::vector<Real> SimulationBox::flattenForces()
{
    if (_forces.size() != _atoms.size() * 3)
        _forces.resize(_atoms.size() * 3);

    Real *const forces = _forces.data();

    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            forces[i * 3 + j] = atom->getForce()[j];
    }
    // clang-format on

    return _forces;
}

/**
 * @brief flattens shift forces of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 */
void SimulationBox::flattenShiftForces()
{
    if (_shiftForces.size() != _atoms.size() * 3)
        _shiftForces.resize(_atoms.size() * 3);

    Real *const shiftForces = _shiftForces.data();

    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            shiftForces[i * 3 + j] = atom->getShiftForce()[j];
    }
    // clang-format on
}

/**
 * @brief flattens old positions of each atom into a single vector of Real
 *
 */
void SimulationBox::flattenOldPositions()
{
    if (_oldPos.size() != _atoms.size() * 3)
        _oldPos.resize(_atoms.size() * 3);

    Real *const oldPositions = _oldPos.data();

    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            oldPositions[i * 3 + j] = atom->getPositionOld()[j];
    }
    // clang-format on
}

/**
 * @brief flattens old velocities of each atom into a single vector of Real
 *
 */
void SimulationBox::flattenOldVelocities()
{
    if (_oldVel.size() != _atoms.size() * 3)
        _oldVel.resize(_atoms.size() * 3);

    Real *const oldVelocities = _oldVel.data();

    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            oldVelocities[i * 3 + j] = atom->getVelocityOld()[j];
    }
    // clang-format on
}

/**
 * @brief flattens old forces of each atom into a single vector of Real
 *
 */
void SimulationBox::flattenOldForces()
{
    if (_oldForces.size() != _atoms.size() * 3)
        _oldForces.resize(_atoms.size() * 3);

    Real *const oldForces = _oldForces.data();

    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            oldForces[i * 3 + j] = atom->getForceOld()[j];
    }
    // clang-format on
}

/**
 * @brief flattens masses of each atom into a single vector of Real
 */
void SimulationBox::flattenMasses()
{
    if (_masses.size() != _atoms.size())
        _masses.resize(_atoms.size());

    Real *const masses = _masses.data();

    // clang-format off
    #pragma omp parallel for
    for (size_t i = 0; i < _atoms.size(); ++i)
        masses[i] = _atoms[i]->getMass();
    // clang-format on
}

/**
 * @brief flattens charges of each atom into a single vector of Real
 *
 */
void SimulationBox::flattenCharges()
{
    if (_charges.size() != _atoms.size())
        _charges.resize(_atoms.size());

    Real *const charges = _charges.data();

    // clang-format off
    #pragma omp parallel for
    for (size_t i = 0; i < _atoms.size(); ++i)
        charges[i] = _atoms[i]->getPartialCharge();
    // clang-format on
}

/**
 * @brief flattens atomtypes of each atom into a single vector of size_t
 * integers
 *
 * @details This method is used to flatten the atom types of each atom into a
 * single vector of size_t integers. This is useful when the atom types are
 * needed in a single vector, for example when calculating the Lennard-Jones
 * potential.
 */
void SimulationBox::flattenAtomTypes()
{
    if (_atomTypes.size() != _atoms.size())
        _atomTypes.resize(_atoms.size());

    size_t *const atomTypes = _atomTypes.data();

    // clang-format off
    #pragma omp parallel for
    for (size_t i = 0; i < _atoms.size(); ++i)
        atomTypes[i] = _atoms[i]->getAtomType();
    // clang-format on
}

/**
 * @brief flattens molecule types of each atom into a single vector of size_t
 * integers
 *
 * @details This method is used to flatten the molecule types of each atom into
 * a single vector of size_t integers. This is useful when the molecule types
 * are needed in a single vector, for example when calculating the Lennard-Jones
 * potential.
 */
void SimulationBox::flattenMolTypes()
{
    if (_molTypes.size() != _molecules.size())
        _molTypes.resize(_molecules.size());

    size_t *const molTypes = _molTypes.data();

    // clang-format off
    #pragma omp parallel for
    for (size_t i = 0; i < _molecules.size(); ++i)
        molTypes[i] = _molecules[i].getMoltype();
    // clang-format on
}

/**
 * @brief flattens the internal global VDW types of each atom into a single
 * vector of size_t integers
 *
 * @details This method is used to flatten the internal global VDW types of each
 * atom into a single vector of size_t integers. This is useful when the global
 * VDW types are needed in a single vector, for example when calculating the
 * Lennard-Jones potential.
 */
void SimulationBox::flattenInternalGlobalVDWTypes()
{
    if (_internalGlobalVDWTypes.size() != _atoms.size())
        _internalGlobalVDWTypes.resize(_atoms.size());

    size_t *const internalGlobalVDWTypes = _internalGlobalVDWTypes.data();

    // clang-format off
    #pragma omp parallel for
    for (size_t i = 0; i < _atoms.size(); ++i)
        internalGlobalVDWTypes[i] = _atoms[i]->getInternalGlobalVDWType();
    // clang-format on
}

/**
 * @brief de-flattens positions of each atom from a single vector of doubles
 * into the atom objects
 *
 * @param positions
 */
void SimulationBox::deFlattenPositions()
{
    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setPosition(_pos[i * 3 + j], j);
    }
    // clang-format on
}

/**
 * @brief de-flattens velocities of each atom from a single vector of doubles
 * into the atom objects
 *
 * @param velocities
 */
void SimulationBox::deFlattenVelocities()
{
    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setVelocity(_vel[i * 3 + j], j);
    }
    // clang-format on
}

/**
 * @brief de-flattens forces of each atom from a single vector of doubles into
 * the atom objects
 *
 * @param forces
 */
void SimulationBox::deFlattenForces()
{
    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setForce(_forces[i * 3 + j], j);
    }
    // clang-format on
}

/**
 * @brief de-flattens shift forces of each atom from a single vector of doubles
 * into the atom objects
 *
 */
void SimulationBox::deFlattenShiftForces()
{
    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setShiftForce(_shiftForces[i * 3 + j], j);
    }
    // clang-format on
}

/**
 * @brief de-flattens old positions of each atom from a single vector of doubles
 * into the atom objects
 *
 */
void SimulationBox::deFlattenOldPositions()
{
    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setPositionOld(_oldPos[i * 3 + j], j);
    }
    // clang-format on
}

/**
 * @brief de-flattens old velocities of each atom from a single vector of
 * doubles into the atom objects
 *
 */
void SimulationBox::deFlattenOldVelocities()
{
    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setVelocityOld(_oldVel[i * 3 + j], j);
    }
    // clang-format on
}

/**
 * @brief de-flattens old forces of each atom from a single vector of doubles
 * into the atom objects
 *
 */
void SimulationBox::deFlattenOldForces()
{
    // clang-format off
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setForceOld(_oldForces[i * 3 + j], j);
    }
    // clang-format on
}

/**
 * @brief reset forces of all atoms
 *
 */
void SimulationBox::resetForces()
{
    flattenForces();

    auto *const _forcesPtr = getForcesPtr();

    // clang-format off

    #pragma omp target teams distribute parallel for is_device_ptr(_forcesPtr)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            _forcesPtr[i * 3 + j] = 0.0;

    // clang-format on

    deFlattenForces();
}

/**
 * @brief calculate center of mass of simulationBox
 *
 * @return Vec3D center of mass
 */
Vec3D SimulationBox::calculateCenterOfMass()
{
    auto comX = 0.0;
    auto comY = 0.0;
    auto comZ = 0.0;

    flattenPositions();

    const auto *const posPtr    = getPosPtr();
    const auto *const massesPtr = getMassesPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for \
                is_device_ptr(posPtr, massesPtr)       \
                map(comX, comY, comZ)                  \
                reduction(+:comX, comY, comZ)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
    {
        comX += massesPtr[i] * posPtr[i * 3];
        comY += massesPtr[i] * posPtr[i * 3 + 1];
        comZ += massesPtr[i] * posPtr[i * 3 + 2];
    }

    _centerOfMass = Vec3D{comX, comY, comZ} / _totalMass;

    return _centerOfMass;
}

/**
 * @brief calculate temperature of simulationBox
 *
 */
double SimulationBox::calculateTemperature()
{
    auto temperature = 0.0;

    flattenVelocities();

    const auto *const velPtr  = getVelPtr();
    const auto *const massPtr = getMassesPtr();

    // clang-format off

    #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(velPtr,massPtr)                    \
                reduction(+:temperature)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            temperature += massPtr[i] * velPtr[i * 3 + j] * velPtr[i * 3 + j];

    // clang-format on

    temperature *= _TEMPERATURE_FACTOR_ / double(_degreesOfFreedom);

    return temperature;
}

/**
 * @brief calculate momentum of simulationBox
 *
 * @return Vec3D
 */
Vec3D SimulationBox::calculateMomentum()
{
    auto momentumX = 0.0;
    auto momentumY = 0.0;
    auto momentumZ = 0.0;

    flattenVelocities();
    const auto *const velPtr    = getVelPtr();
    const auto *const massesPtr = getMassesPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for         \
                is_device_ptr(velPtr, massesPtr)             \
                map(momentumX, momentumY, momentumZ)         \
                reduction(+:momentumX, momentumY, momentumZ)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
    {
        momentumX += massesPtr[i] * velPtr[i * 3 ];
        momentumY += massesPtr[i] * velPtr[i * 3  + 1];
        momentumZ += massesPtr[i] * velPtr[i * 3  + 2];
    }

    // clang-format on

    return Vec3D{momentumX, momentumY, momentumZ};
}

/**
 * @brief calculate angular momentum of simulationBox
 *
 */
Vec3D SimulationBox::calculateAngularMomentum(const Vec3D &momentum)
{
    Real angularMomX = 0.0;
    Real angularMomY = 0.0;
    Real angularMomZ = 0.0;

    flattenPositions();
    flattenVelocities();

    const auto *const massesPtr = getMassesPtr();
    const auto *const posPtr    = getPosPtr();
    const auto *const velPtr    = getVelPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for                \
                is_device_ptr(massesPtr, posPtr, velPtr)            \
                map(angularMomX, angularMomY, angularMomZ)          \
                reduction(+:angularMomX, angularMomY, angularMomZ)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
    {
        const auto mass = massesPtr[i];
        const auto pos  = Vec3D{posPtr[i * 3], posPtr[i * 3 + 1], posPtr[i * 3 + 2]};
        const auto vel  = Vec3D{velPtr[i * 3], velPtr[i * 3 + 1], velPtr[i * 3 + 2]};

        const auto _angularMom = mass * cross(pos, vel);
        angularMomX += _angularMom[0];
        angularMomY += _angularMom[1];
        angularMomZ += _angularMom[2];
    }

    auto angularMom = Vec3D{angularMomX, angularMomY, angularMomZ};

    angularMom -= cross(_centerOfMass, momentum / _totalMass) * _totalMass;

    return angularMom;
}

/**
 * @brief scale all velocities by a factor
 *
 * @param lambda
 */
void SimulationBox::scaleVelocities(const Real lambda)
{
    flattenVelocities();

    Real *const _velPtr = getVelPtr();

    // clang-format off

    #pragma omp target teams distribute parallel for \
                collapse(2)                          \
                is_device_ptr(_velPtr)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            _velPtr[i * 3 + j] *= lambda;

    // clang-format on

    deFlattenVelocities();
}

/**
 * @brief add to all velocities a vector
 *
 * @param velocity
 */
void SimulationBox::addToVelocities(const Vec3D &velocity)
{
    flattenVelocities();

    Real *const _velPtr = getVelPtr();

    // clang-format off

    #pragma omp target teams distribute parallel for \
                collapse(2)                          \
                is_device_ptr(_velPtr)               \
                map(velocity)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            _velPtr[i * 3 + j] += velocity[j];

    // clang-format on

    deFlattenVelocities();
}

/**
 * @brief get atomic scalar forces
 *
 * @return std::vector<Real>
 */
std::vector<Real> SimulationBox::getAtomicScalarForces()
{
    flattenForces();

    const auto *const forcesPtr = getForcesPtr();

    const auto nAtoms = getNumberOfAtoms();

    std::vector<Real> atomicScalarForces;
    atomicScalarForces.reserve(nAtoms);

    auto *const scalarForces = atomicScalarForces.data();

    // clang-format off
    #pragma omp target teams distribute parallel for \
                is_device_ptr(forcesPtr)             \
                map(scalarForces)
    for (size_t i = 0; i < nAtoms; ++i)
        scalarForces[i] = ::sqrt(forcesPtr[i * 3]     * forcesPtr[i * 3]     +
                                 forcesPtr[i * 3 + 1] * forcesPtr[i * 3 + 1] +
                                 forcesPtr[i * 3 + 2] * forcesPtr[i * 3 + 2]);
    // clang-format on

    return atomicScalarForces;
}

/**
 * @brief get atomic scalar forces
 *
 * @return std::vector<Real>
 */
std::vector<Real> SimulationBox::getAtomicScalarForcesOld()
{
    flattenOldForces();

    const auto *const forcesPtr = getOldForcesPtr();

    const auto nAtoms = getNumberOfAtoms();

    std::vector<Real> atomicScalarForces;
    atomicScalarForces.reserve(nAtoms);

    auto *const scalarForces = atomicScalarForces.data();

    // clang-format off
    #pragma omp target teams distribute parallel for \
                is_device_ptr(forcesPtr)             \
                map(scalarForces)
    for (size_t i = 0; i < nAtoms; ++i)
        scalarForces[i] = ::sqrt(forcesPtr[i * 3]     * forcesPtr[i * 3]     +
                                 forcesPtr[i * 3 + 1] * forcesPtr[i * 3 + 1] +
                                 forcesPtr[i * 3 + 2] * forcesPtr[i * 3 + 2]);
    // clang-format on

    return atomicScalarForces;
}

/**
 * @brief update old positions of all atoms
 *
 */
void SimulationBox::updateOldPositions()
{
    flattenPositions();

    const auto *const posPtr    = getPosPtr();
    auto *const       oldPosPtr = getOldPosPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for \
                is_device_ptr(posPtr, oldPosPtr)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            oldPosPtr[i * 3 + j] = posPtr[i * 3 + j];
    // clang-format on

    deFlattenOldPositions();
}

/**
 * @brief update old velocities of all atoms
 *
 */
void SimulationBox::updateOldVelocities()
{
    flattenVelocities();

    const auto *const velPtr    = getVelPtr();
    auto *const       oldVelPtr = getOldVelPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for \
                is_device_ptr(velPtr, oldVelPtr)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            oldVelPtr[i * 3 + j] = velPtr[i * 3 + j];
    // clang-format on

    deFlattenOldVelocities();
}

/**
 * @brief update old forces of all atoms
 *
 */
void SimulationBox::updateOldForces()
{
    flattenForces();

    const auto *const forcesPtr    = getForcesPtr();
    auto *const       oldForcesPtr = getOldForcesPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for \
                is_device_ptr(forcesPtr, oldForcesPtr)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            oldForcesPtr[i * 3 + j] = forcesPtr[i * 3 + j];
    // clang-format on

    deFlattenOldForces();
}

/**
 * @brief initializes a vector that is n molecules long and each element
 * contains the number of atoms in the molecule
 *
 */
void SimulationBox::initAtomsPerMolecule()
{
    _atomsPerMolecule.resize(_molecules.size());

    // clang-format off
    #pragma omp parallel for
    for (size_t i = 0; i < _molecules.size(); ++i)
        _atomsPerMolecule[i] = _molecules[i].getNumberOfAtoms();
    // clang-format on
}

/**
 * @brief initializes a vector that is n atoms long and each element contains
 * the index of the molecule to which the atom belongs
 *
 * @details This method is used to initialize a vector that is n atoms long and
 * each element contains the index of the molecule to which the atom belongs.
 * This is useful when the atom types are needed in a single vector, for example
 * when calculating the Lennard-Jones potential.
 */
void SimulationBox::initMoleculeIndices()
{
    _moleculeIndices.resize(_atoms.size());

    size_t moleculeIndex = 0;

    for (size_t i = 0; i < _molecules.size(); ++i)
    {
        const auto nAtoms = _molecules[i].getNumberOfAtoms();

        for (size_t j = 0; j < nAtoms; ++j)
            _moleculeIndices[moleculeIndex++] = i;
    }
}