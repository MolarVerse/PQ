#ifndef _RST_FILE_OUTPUT_HPP_

#define _RST_FILE_OUTPUT_HPP_

#include "output.hpp"
#include "simulationBox.hpp"

namespace output
{
    class RstFileOutput;
}

/**
 * @class RstFileOutput inherits from Output
 *
 * @brief Output file for restart file
 *
 */
class output::RstFileOutput : public output::Output
{
  public:
    using output::Output::Output;

    void write(simulationBox::SimulationBox &, const size_t);
};

#endif   // _RST_FILE_OUTPUT_HPP_