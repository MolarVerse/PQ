#include "molecule.hpp"
#include "exceptions.hpp"

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
void Molecule::setNumberOfAtoms(const size_t numberOfAtoms)
{
    _numberOfAtoms = numberOfAtoms;
}

/**
 * @brief adds atomic vector in xyz format to molecule
 *
 * @param v
 * @param vToAdd
 */
void Molecule::addVector(vector<double> &v, const vector<double> &vToAdd) const
{
    v.insert(v.end(), vToAdd.begin(), vToAdd.end());
}

/**
 * @brief finds number of different atom types in molecule
 *
 * @return int
 */
size_t Molecule::getNumberOfAtomTypes()
{
    return static_cast<size_t>(distance(_externalAtomTypes.begin(), unique(_externalAtomTypes.begin(), _externalAtomTypes.end())));
}

void Molecule::calculateCenterOfMass(const vector<double> &box)
{
    _centerOfMass = vector<double>(3, 0.0);

    auto positionAtom1 = getAtomPositions(0);

    for (size_t i = 0; i < _numberOfAtoms; ++i)
    {
        const auto mass = getMass(i);
        auto position = getAtomPositions(i);

        _centerOfMass[0] += mass * (position[0] - box[0] * round((position[0] - positionAtom1[0]) / box[0]));
        _centerOfMass[1] += mass * (position[1] - box[1] * round((position[1] - positionAtom1[1]) / box[1]));
        _centerOfMass[2] += mass * (position[2] - box[2] * round((position[2] - positionAtom1[2]) / box[2]));
    }

    _centerOfMass[0] /= getMolMass();
    _centerOfMass[1] /= getMolMass();
    _centerOfMass[2] /= getMolMass();
}