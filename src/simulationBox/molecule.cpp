#include "molecule.hpp"
#include "exceptions.hpp"

void Molecule::setNumberOfAtoms(int numberOfAtoms)
{
    if (numberOfAtoms < 0)
    {
        throw MolDescriptorException("Number of atoms in molecule " + _name + " is negative");
    }
    _numberOfAtoms = numberOfAtoms;
}