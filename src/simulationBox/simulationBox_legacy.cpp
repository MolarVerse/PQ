#include "constants.hpp"
#include "simulationBox.hpp"

using namespace simulationBox;
using namespace constants;

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