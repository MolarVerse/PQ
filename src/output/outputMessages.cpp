#include "outputMessages.hpp"

#include <format>   // for format
#include <string>   // for string

/**
 * @brief construct general initial momentum message
 *
 * @param momentum
 * @return string
 */
std::string output::initialMomentumMessage(const double initialMomentum)
{
    return std::format("Initial momentum = {} Angstrom * amu / fs", initialMomentum);
}