#ifndef _STDOUT_OUTPUT_HPP_

#define _STDOUT_OUTPUT_HPP_

#include "output.hpp"

namespace output
{
    class StdoutOutput;
}

/**
 * @class StdoutOutput inherits from Output
 *
 * @brief Output file for stdout
 *
 */
class output::StdoutOutput : public output::Output
{
  public:
    using output::Output::Output;

    void writeDensityWarning() const;
    void writeInitialMomentum(const double momentum) const;
};

#endif   // _STDOUT_OUTPUT_HPP_