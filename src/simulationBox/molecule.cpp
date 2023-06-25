#include "molecule.hpp"

#include "atomMassMap.hpp"
#include "atomNumberMap.hpp"
#include "exceptions.hpp"
#include "vector3d.hpp"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace simulationBox;
using namespace vector3d;
using namespace config;

/**
 * @brief sets number of atoms in molecule
 *
 * @param numberOfAtoms
 *
 * TODO: add check for number of atoms when reading moldescriptor if not negative
 */

/**
 * @brief finds number of different atom types in molecule
 *
 * @return int
 */
size_t Molecule::getNumberOfAtomTypes()
{
    return static_cast<size_t>(
        distance(_externalAtomTypes.begin(), unique(_externalAtomTypes.begin(), _externalAtomTypes.end())));
}

/**
 * @brief calculates the center of mass of the molecule
 *
 * @param box
 */
void Molecule::calculateCenterOfMass(const Vec3D &box)
{
    _centerOfMass            = Vec3D();
    const auto positionAtom1 = getAtomPosition(0);

    for (size_t i = 0; i < _numberOfAtoms; ++i)
    {
        const auto mass     = getAtomMass(i);
        const auto position = getAtomPosition(i);

        _centerOfMass += mass * (position - box * round((position - positionAtom1) / box));   // PBC to first atom of molecule
    }

    _centerOfMass /= getMolMass();
}

/**
 * @brief scales the positions of the molecule by shifting the center of mass
 *
 * @param shiftFactors
 */
void Molecule::scale(const Vec3D &shiftFactors)
{
    const auto shift = _centerOfMass * (shiftFactors - 1.0);

    for (size_t i = 0; i < _numberOfAtoms; ++i)
        _positions[i] += shift;
}

/**
 * @brief scales the velocities of the molecule
 *
 * @param scaleFactor
 */
void Molecule::scaleVelocities(double scaleFactor)
{
    for (size_t i = 0; i < _numberOfAtoms; ++i)
        _velocities[i] *= scaleFactor;
}