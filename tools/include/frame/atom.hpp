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