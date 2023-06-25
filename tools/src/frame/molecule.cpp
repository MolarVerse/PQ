#include "molecule.hpp"

#include "atomMassMap.hpp"

#include <iostream>

using namespace std;
using namespace frameTools;
using namespace vector3d;
using namespace config;

/**
 * @brief calculates the center of mass of the molecule
 *
 * @param box
 */
void Molecule::calculateCenterOfMass(const Vec3D &box)
{
    _molMass = 0.0;

    const auto xyz_0 = _atoms[0]->getPosition();

    for (auto atom : _atoms)
    {
        const auto   atomName = atom->getElementType();
        const double mass     = atomMassMap.at(atomName);
        auto         position = atom->getPosition();

        position      -= box * round((position - xyz_0) / box);
        _centerOfMass += mass * position;

        _molMass += mass;
    }

    _centerOfMass /= _molMass;
}