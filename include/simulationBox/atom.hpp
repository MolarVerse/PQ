#ifndef _ATOM_HPP_

#define _ATOM_HPP_

#include "vector3d.hpp"   // for Vec3D

#include <string>   // for string

namespace simulationBox
{
    /**
     * @class Atom
     *
     * @brief containing all information about an atom
     */
    class Atom
    {
      private:
        std::string _name;

        int    _atomicNumber;
        double _mass;

        linearAlgebra::Vec3D _position;
        linearAlgebra::Vec3D _velocity;
        linearAlgebra::Vec3D _force;

      public:
        Atom() = default;
        explicit Atom(const std::string_view &name) : _name(name){};

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] std::string getName() const { return _name; }

        [[nodiscard]] linearAlgebra::Vec3D getPosition() const { return _position; }
        [[nodiscard]] linearAlgebra::Vec3D getVelocity() const { return _velocity; }
        [[nodiscard]] linearAlgebra::Vec3D getForce() const { return _force; }

        /***************************
         * standard setter methods *
         ***************************/

        void setPosition(const linearAlgebra::Vec3D &position) { _position = position; }
        void setVelocity(const linearAlgebra::Vec3D &velocity) { _velocity = velocity; }
        void setForce(const linearAlgebra::Vec3D &force) { _force = force; }
    };
}   // namespace simulationBox

#endif   // _ATOM_HPP_