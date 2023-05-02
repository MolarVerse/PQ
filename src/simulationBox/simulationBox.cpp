#include "simulationBox.hpp"

using namespace std;

/**
 * @brief sets the atomic 3d properties
 *
 * @param target
 * @param toAdd
 *
 * @details
 *
 *  The toAdd vector entries are stored in a vector of doubles.
 *  The vector is flattened, i.e. the first three elements
 *  are the x, y and z coordinates of the first atom,
 *  the next three elements are the x, y and z coordinates
 *  of the second atom, and so on.
 */
void SimulationBox::setAtomicProperties(vector<double> &target, vector<double> toAdd) const
{
    for (auto toAddElement : toAdd)
        target.push_back(toAddElement);
}