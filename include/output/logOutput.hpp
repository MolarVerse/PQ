#ifndef _LOG_OUTPUT_HPP_

#define _LOG_OUTPUT_HPP_

#include "output.hpp"

namespace output
{
    /**
     * @class LogOutput inherits from Output
     *
     * @brief Output file for log file
     *
     */
    class LogOutput : public Output
    {
      public:
        using Output::Output;

        void writeHeader();
        void writeDensityWarning();
        void writeInitialMomentum(const double momentum);
    };

}   // namespace output

#endif   // _LOG_OUTPUT_HPP_