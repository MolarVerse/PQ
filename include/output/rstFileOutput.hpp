#ifndef _RST_FILE_OUTPUT_HPP_

#define _RST_FILE_OUTPUT_HPP_

#include "output.hpp"

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
};

#endif   // _RST_FILE_OUTPUT_HPP_