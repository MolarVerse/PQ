#ifndef _TYPE_ALIASES_HPP_

#define _TYPE_ALIASES_HPP_

#include <cstddef>      // for size_t
#include <functional>   // for std::function
#include <memory>       // for std::shared_ptr
#include <string>       // for std::string
#include <vector>       // for std::vector

#include "staticMatrix3x3Class.hpp"
#include "vector3d.hpp"

namespace simulationBox
{
    class SimulationBox;   // forward declaration

}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration

}   // namespace physicalData

namespace pq
{
    using strings = std::vector<std::string>;

    using Vec3D    = linearAlgebra::Vec3D;
    using tensor3D = linearAlgebra::tensor3D;

    using SharedSimulationBox = std::shared_ptr<simulationBox::SimulationBox>;
    using SharedPhysicalData  = std::shared_ptr<physicalData::PhysicalData>;

}   // namespace pq

#endif   // _TYPE_ALIASES_HPP_