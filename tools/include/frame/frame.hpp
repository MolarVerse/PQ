#ifndef _FRAME_HPP_

#define _FRAME_HPP_

#include "atom.hpp"
#include "molecule.hpp"

#include <string>
#include <vector>

namespace frameTools
{
    class Frame
    {
      private:
        size_t          _nAtoms;
        vector3d::Vec3D _box;

        std::vector<Atom>     _atoms;
        std::vector<Molecule> _molecules;

      public:
        std::string getElementType(const size_t atomIndex) const { return _atoms[atomIndex].getElementType(); }

        vector3d::Vec3D getPosition(const size_t atomIndex) const { return _atoms[atomIndex].getPosition(); }

        // standard getter and setter
        void   setNAtoms(const size_t nAtoms) { _nAtoms = nAtoms; }
        size_t getNAtoms() const { return _nAtoms; }

        void setBox(const vector3d::Vec3D &box) { _box = box; }

        void              addAtom(const Atom &atom) { _atoms.push_back(atom); }
        Atom             &getAtom(const size_t atomIndex) { return _atoms[atomIndex]; }
        std::vector<Atom> getAtoms() const { return _atoms; }

        vector3d::Vec3D getBox() const { return _box; }

        void                   addMolecule(const Molecule &molecule) { _molecules.push_back(molecule); }
        std::vector<Molecule> &getMolecules() { return _molecules; }
    };
}   // namespace frameTools

#endif   // _FRAME_HPP_