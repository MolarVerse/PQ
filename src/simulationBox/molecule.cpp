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
size_t Molecule::getNumberOfAtomTypes() { return _externalAtomTypes.size() - ranges::size(ranges::unique(_externalAtomTypes)); }

/**
 * @brief calculates the center of mass of the molecule
 *
 * @param box
 */
void Molecule::calculateCenterOfMass(const Vec3D &box)
{
    _centerOfMass            = Vec3D();
    const auto positionAtom1 = getAtomPosition(0);

    // TODO: sonarlint until now not compatible with c++23
    //  auto f = [&_centerOfMass = _centerOfMass, &positionAtom1, &box = box](auto &&pair)
    //  {
    //      auto const &[mass, position]  = pair;
    //      _centerOfMass                += mass * (position - box * round((position - positionAtom1) / box));
    //  };
    //  ranges::for_each(ranges::views::zip(_masses, _positions), f);

    for (size_t i = 0; i < _numberOfAtoms; ++i)
    {
        const auto mass     = _masses[i];
        const auto position = _positions[i];

        _centerOfMass += mass * (position - box * round((position - positionAtom1) / box));
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

    ranges::for_each(_positions, [shift](auto &position) { position += shift; });
}

/**
 * @brief scales the velocities of the molecule
 *
 * @param scaleFactor
 */
void Molecule::scaleVelocities(const double scaleFactor)
{
    ranges::for_each(_velocities, [scaleFactor](auto &velocity) { velocity *= scaleFactor; });
}

/**
 * @brief corrects the velocities of the molecule
 *
 * @param correction
 */
void Molecule::correctVelocities(const Vec3D &correction)
{
    ranges::for_each(_velocities, [correction](auto &velocity) { velocity -= correction; });
}