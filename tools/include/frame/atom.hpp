#ifndef _ATOMS_HPP_

#define _ATOMS_HPP_

#include <string>

#include "vector3d.hpp"

namespace frameTools
{
    class Atom
    {
    private:
        std::string _atomName;
        std::string _elementType;

        Vec3D _position;
        Vec3D _velocity;
        Vec3D _force;

    public:
        Atom() = default;
        explicit Atom(const std::string &atomName);

        // standard getter and setter
        [[nodiscard]] std::string getElementType() const { return _elementType; }

        void setPosition(const Vec3D &position) { _position = position; }
        [[nodiscard]] Vec3D getPosition() const { return _position; }
    };
}

#endif // _ATOMS_HPP_