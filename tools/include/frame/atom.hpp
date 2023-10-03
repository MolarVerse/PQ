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

#ifndef _ATOMS_HPP_

#define _ATOMS_HPP_

#include "vector3d.hpp"

#include <string>

namespace frameTools
{
    class Atom
    {
      private:
        std::string _atomName;
        std::string _elementType;

        linearAlgebra::Vec3D _position;
        linearAlgebra::Vec3D _velocity;
        linearAlgebra::Vec3D _force;

      public:
        Atom() = default;
        explicit Atom(const std::string &atomName);

        // standard getter and setter
        std::string getElementType() const { return _elementType; }

        void                               setPosition(const linearAlgebra::Vec3D &position) { _position = position; }
        [[nodiscard]] linearAlgebra::Vec3D getPosition() const { return _position; }
    };
}   // namespace frameTools

#endif   // _ATOMS_HPP_