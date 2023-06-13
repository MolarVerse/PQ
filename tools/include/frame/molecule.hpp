#ifndef _MOLECULE_HPP_

#define _MOLECULE_HPP_

#include <vector>

#include "atom.hpp"

class Molecule
{
private:
    size_t _nAtoms;
    std::vector<Atom> _atoms;

public:
    Molecule() = default;
    explicit Molecule(const size_t nAtoms);

    void addAtom(const Atom &atom) { _atoms.push_back(atom); }
};

#endif // _MOLECULE_HPP_