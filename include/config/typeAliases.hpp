#ifndef _TYPE_ALIASES_HPP_

#define _TYPE_ALIASES_HPP_

#include <cstddef>      // for size_t
#include <functional>   // for std::function
#include <string>       // for std::string
#include <vector>       // for std::vector

#include "matrix.hpp"            // for tensor3D
#include "staticMatrix3x3.hpp"   // for StaticMatrix3x3
#include "vector3d.hpp"          // for Vec3D

namespace pq
{
    using strings = std::vector<std::string>;

    using Vec3D            = linearAlgebra::Vec3D;
    using tensor3D         = linearAlgebra::tensor3D;
    using StaticMatrix3x3D = linearAlgebra::StaticMatrix3x3<double>;

}   // namespace pq

#endif   // _TYPE_ALIASES_HPP_