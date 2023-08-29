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
        size_t               _nAtoms;
        linearAlgebra::Vec3D _box;

        std::vector<Atom>     _atoms;
        std::vector<Molecule> _molecules;

      public:
        std::string getElementType(const size_t atomIndex) const { return _atoms[atomIndex].getElementType(); }

        [[nodiscard]] linearAlgebra::Vec3D getPosition(const size_t atomIndex) const { return _atoms[atomIndex].getPosition(); }

        // standard getter and setter
        void                 setNAtoms(const size_t nAtoms) { _nAtoms = nAtoms; }
        [[nodiscard]] size_t getNAtoms() const { return _nAtoms; }

        void setBox(const linearAlgebra::Vec3D &box) { _box = box; }

        void                            addAtom(const Atom &atom) { _atoms.push_back(atom); }
        [[nodiscard]] Atom             &getAtom(const size_t atomIndex) { return _atoms[atomIndex]; }
        [[nodiscard]] std::vector<Atom> getAtoms() const { return _atoms; }

        [[nodiscard]] linearAlgebra::Vec3D getBox() const { return _box; }

        void                                 addMolecule(const Molecule &molecule) { _molecules.push_back(molecule); }
        [[nodiscard]] std::vector<Molecule> &getMolecules() { return _molecules; }
    };
}   // namespace frameTools

#endif   // _FRAME_HPP_