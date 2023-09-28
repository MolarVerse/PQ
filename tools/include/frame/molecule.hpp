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

#ifndef _MOLECULE_HPP_

#define _MOLECULE_HPP_

#include "vector3d.hpp"   // for Vec3D

#include <cstddef>   // for size_t
#include <vector>

namespace frameTools
{
    class Atom;   // forward declaration
}

namespace frameTools
{
    class Molecule
    {
      private:
        size_t _nAtoms;

        double               _molMass      = 0.0;
        linearAlgebra::Vec3D _centerOfMass = {0.0, 0.0, 0.0};

        std::vector<Atom *> _atoms;

      public:
        Molecule() = default;
        explicit Molecule(const size_t nAtoms) : _nAtoms(nAtoms) {}

        void calculateCenterOfMass(const linearAlgebra::Vec3D &);

        linearAlgebra::Vec3D getCenterOfMass() const { return _centerOfMass; }

        void addAtom(Atom *atom) { _atoms.push_back(atom); }
    };

}   // namespace frameTools

#endif   // _MOLECULE_HPP_