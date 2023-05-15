#include "molecule.hpp"
#include "exceptions.hpp"

#include <algorithm>

using namespace std;

/**
 * @brief sets number of atoms in molecule
 *
 * @param numberOfAtoms
 *
 * @throw MolDescriptorException if number of atoms is negative
 */
void Molecule::setNumberOfAtoms(int numberOfAtoms)
{
    if (numberOfAtoms < 0)
    {
        throw MolDescriptorException("Number of atoms in molecule " + _name + " is negative");
    }
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
    for (auto element : vToAdd)
    {
        v.push_back(element);
    }
}

/**
 * @brief finds number of different atom types in molecule
 *
 * @return int
 */
int Molecule::getNumberOfAtomTypes()
{
    return int(distance(_externalAtomTypes.begin(), unique(_externalAtomTypes.begin(), _externalAtomTypes.end())));
}