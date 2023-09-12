#include "molecule.hpp"

#include "atom.hpp"          // for Atom, frameTools
#include "atomMassMap.hpp"   // for atomMassMap

#include <iostream>   // for std
#include <map>        // for map
#include <string>     // for string, operator<=>

using namespace std;
using namespace frameTools;
using namespace linearAlgebra;

/**
 * @brief calculates the center of mass of the molecule
 *
 * @param box
 */
void Molecule::calculateCenterOfMass(const Vec3D &box)
{
    _molMass = 0.0;

    const auto xyz_0 = _atoms[0]->getPosition();

    for (const auto *atom : _atoms)
    {
        const auto   atomName = atom->getElementType();
        const double mass     = constants::atomMassMap.at(atomName);
        auto         position = atom->getPosition();

        position      -= box * round((position - xyz_0) / box);
        _centerOfMass += mass * position;

        _molMass += mass;
    }

    _centerOfMass /= _molMass;
}