#include "constants.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"

using namespace simulationBox;
using namespace constants;
using namespace settings;

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
 *
 * @return std::vector<Real>
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
 * @brief calculate temperature of simulationBox
 *
 */
double SimulationBox::calculateTemperature()
{
    auto temperature = 0.0;

    flattenVelocities();

    const auto *const _velPtr  = getVelPtr();
    const auto *const _massPtr = getMassesPtr();

    // clang-format off

    #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(_velPtr)                           \
                reduction(+:temperature)
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            temperature += _massPtr[i] * _velPtr[i * 3 + j] * _velPtr[i * 3 + j];

    // clang-format on

    temperature *= _TEMPERATURE_FACTOR_ / double(_degreesOfFreedom);

    deFlattenVelocities();

    return temperature;
}