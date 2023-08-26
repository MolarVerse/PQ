#include "molecule.hpp"

#include "vector3d.hpp"

#include <algorithm>    // for std::ranges::for_each
#include <functional>   // for identity, equal_to
#include <iterator>     // for _Size, size
#include <ranges>       // for subrange

using namespace simulationBox;

/**
 * @brief finds number of different atom types in molecule
 *
 * @return int
 */
size_t Molecule::getNumberOfAtomTypes()
{
    return _externalAtomTypes.size() - std::ranges::size(std::ranges::unique(_externalAtomTypes));
}

/**
 * @brief calculates the center of mass of the molecule
 *
 * @details distances are calculated relative to the first atom
 *
 * @param box
 */
void Molecule::calculateCenterOfMass(const linearAlgebra::Vec3D &box)
{
    _centerOfMass            = linearAlgebra::Vec3D();
    const auto positionAtom1 = getAtomPosition(0);

    // TODO: sonarlint until now not compatible with c++23
    //  auto f = [&_centerOfMass = _centerOfMass, &positionAtom1, &box = box](auto &&pair)
    //  {
    //      auto const &[mass, position]  = pair;
    //      _centerOfMass                += mass * (position - box * round((position - positionAtom1) / box));
    //  };
    //  std::ranges::for_each(std::ranges::views::zip(_masses, _positions), f);

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
void Molecule::scale(const linearAlgebra::Vec3D &shiftFactors)
{
    const auto shift = _centerOfMass * (shiftFactors - 1.0);

    std::ranges::for_each(_positions, [shift](auto &position) { position += shift; });
}

/**
 * @brief scales the velocities of the molecule with a multiplicative factor
 *
 * @param scaleFactor
 */
void Molecule::scaleVelocities(const double scaleFactor)
{
    std::ranges::for_each(_velocities, [scaleFactor](auto &velocity) { velocity *= scaleFactor; });
}

/**
 * @brief corrects the velocities of the molecule by a given shift vector
 *
 * @param correction
 */
void Molecule::correctVelocities(const linearAlgebra::Vec3D &correction)
{
    std::ranges::for_each(_velocities, [correction](auto &velocity) { velocity -= correction; });
}