#include "molecule.hpp"
#include "exceptions.hpp"
#include "vector3d.hpp"

#include <algorithm>
#include <cmath>

using namespace std;

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
size_t
Molecule::getNumberOfAtomTypes()
{
    return static_cast<size_t>(distance(_externalAtomTypes.begin(), unique(_externalAtomTypes.begin(), _externalAtomTypes.end())));
}

void Molecule::calculateCenterOfMass(const Vec3D &box)
{
    _centerOfMass = Vec3D();

    const auto positionAtom1 = getAtomPositions(0);

    for (size_t i = 0; i < _numberOfAtoms; ++i)
    {
        const auto mass = getMass(i);
        const auto position = getAtomPositions(i);

        _centerOfMass += mass * (position - box * round((position - positionAtom1) / box));
    }

    _centerOfMass /= getMolMass();
}