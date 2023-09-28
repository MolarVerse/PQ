/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _FRAME_HPP_

#define _FRAME_HPP_

#include "atom.hpp"       // for Atom
#include "molecule.hpp"   // for Molecule
#include "vector3d.hpp"   // for Vec3D

#include <stddef.h>   // for size_t
#include <string>     // for string
#include <vector>     // for vector

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