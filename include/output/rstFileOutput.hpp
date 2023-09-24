#ifndef _RST_FILE_OUTPUT_HPP_

#define _RST_FILE_OUTPUT_HPP_

#include "output.hpp"

#include <cstddef>   // for size_t

namespace simulationBox
{
    class SimulationBox; // forward declaration
}

namespace output
{
    /**
     * @class RstFileOutput inherits from Output
     *
     * @brief Output file for restart file
     *
     */
    class RstFileOutput : public Output
    {
      public:
        using Output::Output;

        void write(simulationBox::SimulationBox &, const size_t);
    };

}   // namespace output

#endif   // _RST_FILE_OUTPUT_HPP_