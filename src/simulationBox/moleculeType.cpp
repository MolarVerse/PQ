#include "moleculeType.hpp"

#include <algorithm>   // for sort, unique

using namespace simulationBox;

/**
 * @brief finds number of different atom types in molecule
 *
 * @return int
 */
size_t MoleculeType::getNumberOfAtomTypes()
{
    return _externalAtomTypes.size() - std::ranges::size(std::ranges::unique(_externalAtomTypes));
}