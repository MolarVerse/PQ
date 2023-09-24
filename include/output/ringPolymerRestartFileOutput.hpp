#ifndef _RING_POLYMER_RESTART_FILE_OUTPUT_HPP_

#define _RING_POLYMER_RESTART_FILE_OUTPUT_HPP_

#include "output.hpp"

#include <cstddef>   // for size_t
#include <vector>    // for vector

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace output
{
    /**
     * @class RingPolymerRestartFileOutput inherits from Output
     *
     * @brief Output file for restart file
     *
     */
    class RingPolymerRestartFileOutput : public Output
    {
      public:
        using Output::Output;

        void write(std::vector<simulationBox::SimulationBox> &, const size_t);
    };

}   // namespace output

#endif   // _RST_FILE_OUTPUT_HPP_