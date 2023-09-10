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

        linearAlgebra::Vec3D _positions;
        linearAlgebra::Vec3D _velocities;
        linearAlgebra::Vec3D _forces;
    };
}   // namespace simulationBox

#endif   // _ATOM_HPP_