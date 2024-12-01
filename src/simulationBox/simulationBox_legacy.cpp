#include <algorithm>
#include <ranges>

#include "constants.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

using namespace simulationBox;
using namespace constants;
using namespace linearAlgebra;

/**
 * @brief calculate total mass of simulationBox
 *
 */
void SimulationBox::calculateTotalMass()
{
    _totalMass = 0.0;

    auto accumulateMass = [this](const auto& atom)
    { _totalMass += atom->getMass(); };

    std::ranges::for_each(_atoms, accumulateMass);
}

/**
 * @brief calculate total force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateTotalForce()
{
    Vec3D totalForce(0.0);

    auto accumulateForce = [&totalForce](const auto& atom)
    { totalForce += atom->getForce(); };

    std::ranges::for_each(_atoms, accumulateForce);

    return norm(totalForce);
}

/**
 * @brief flattens positions of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::flattenPositions()
{
    std::vector<double> positions;

    auto addPositions = [&positions](auto& atom)
    {
        const auto position = atom->getPosition();

        positions.push_back(position[0]);
        positions.push_back(position[1]);
        positions.push_back(position[2]);
    };

    std::ranges::for_each(_atoms, addPositions);

    return positions;
}

/**
 * @brief reset forces of all atoms
 *
 */
void SimulationBox::resetForces()
{
    auto resetForce = [](const auto& atom) { atom->setForceToZero(); };

    std::ranges::for_each(_atoms, resetForce);
}

/**
 * @brief calculate center of mass of simulationBox
 *
 * @return Vec3D center of mass
 */
Vec3D SimulationBox::calculateCenterOfMass()
{
    _centerOfMass = Vec3D{0.0};

    auto accumulateMassWeightedPos = [this](const auto& atom)
    { _centerOfMass += atom->getMass() * atom->getPosition(); };

    std::ranges::for_each(_atoms, accumulateMassWeightedPos);

    _centerOfMass /= _totalMass;

    return _centerOfMass;
}

/**
 * @brief calculate temperature of simulationBox
 *
 */
double SimulationBox::calculateTemperature()
{
    auto temperature = 0.0;

    auto accumulateTemperature = [&temperature](const auto& atom)
    { temperature += atom->getMass() * normSquared(atom->getVelocity()); };

    std::ranges::for_each(_atoms, accumulateTemperature);

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
    auto momentum = Vec3D{0.0};

    auto accumulateAtomicMomentum = [&momentum](const auto& atom)
    { momentum += atom->getMass() * atom->getVelocity(); };

    std::ranges::for_each(_atoms, accumulateAtomicMomentum);

    return momentum;
}

/**
 * @brief calculate angular momentum of simulationBox
 *
 */
Vec3D SimulationBox::calculateAngularMomentum(const Vec3D& momentum)
{
    auto angularMom = Vec3D{0.0};

    auto accumulateAngularMomentum = [&angularMom](const auto& atom)
    {
        const auto mass = atom->getMass();
        angularMom += mass * cross(atom->getPosition(), atom->getVelocity());
    };

    std::ranges::for_each(_atoms, accumulateAngularMomentum);

    angularMom -= cross(_centerOfMass, momentum / _totalMass) * _totalMass;

    return angularMom;
}

/**
 * @brief scale all velocities by a factor
 *
 */
void SimulationBox::scaleVelocities(const double lambda)
{
    auto scaleVelocity = [lambda](auto& atom) { atom->scaleVelocity(lambda); };

    std::ranges::for_each(_atoms, scaleVelocity);
}

/**
 * @brief add to all velocities a vector
 *
 * @param velocity
 */
void SimulationBox::addToVelocities(const Vec3D& velocity)
{
    auto addVelocity = [&velocity](auto& atom) { atom->addVelocity(velocity); };

    std::ranges::for_each(_atoms, addVelocity);
}

/**
 * @brief get atomic scalar forces
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::getAtomicScalarForces() const
{
    std::vector<double> atomicScalarForces;

    for (const auto& atom : _atoms)
        atomicScalarForces.push_back(norm(atom->getForce()));

    return atomicScalarForces;
}

/**
 * @brief get atomic scalar forces old
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::getAtomicScalarForcesOld() const
{
    std::vector<double> atomicScalarForces;

    for (const auto& atom : _atoms)
        atomicScalarForces.push_back(norm(atom->getForceOld()));

    return atomicScalarForces;
}

/**
 * @brief update old positions of all atoms
 *
 */
void SimulationBox::updateOldPositions()
{
    auto updateOldPosition = [](const auto& atom)
    { atom->updateOldPosition(); };

    std::ranges::for_each(_atoms, updateOldPosition);
}

/**
 * @brief update old velocities of all atoms
 *
 */
void SimulationBox::updateOldVelocities()
{
    auto updateOldVelocity = [](const auto& atom)
    { atom->updateOldVelocity(); };

    std::ranges::for_each(_atoms, updateOldVelocity);
}

/**
 * @brief update old forces of all atoms
 *
 */
void SimulationBox::updateOldForces()
{
    auto updateOldForce = [](const auto& atom) { atom->updateOldForce(); };

    std::ranges::for_each(_atoms, updateOldForce);
}