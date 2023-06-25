#ifndef _LOG_OUTPUT_HPP_

#define _LOG_OUTPUT_HPP_

#include "output.hpp"

namespace output
{
    class LogOutput;
}

/**
 * @class LogOutput inherits from Output
 *
 * @brief Output file for log file
 *
 */
class output::LogOutput : public output::Output
{
  public:
    using output::Output::Output;

    void writeDensityWarning();
    void writeRelaxationTimeThermostatWarning();
    void writeRelaxationTimeManostatWarning();
    void writeInitialMomentum(const double momentum);
};

#endif   // _LOG_OUTPUT_HPP_