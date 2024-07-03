#ifndef _SIMULATION_BOX_TPP_

#define _SIMULATION_BOX_TPP_

#include "simulationBox.hpp"

namespace simulationBox
{
    /**
     * @brief set the box depending on dynamic type
     *
     * @tparam T
     * @param box
     */
    template <typename T>
    void SimulationBox::setBox(const T& box)
    {
        _box = std::make_shared<T>(box);
    }

}   // namespace simulationBox

#endif   // _SIMULATION_BOX_TPP_