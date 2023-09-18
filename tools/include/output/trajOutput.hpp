#ifndef _TRAJ_OUTPUT_HPP_

#define _TRAJ_OUTPUT_HPP_

#include "output.hpp"   // for Output

namespace frameTools
{
    class Frame;   // forward declaration
}

class TrajOutput : public output::Output
{
  public:
    using output::Output::Output;

    void write(frameTools::Frame &);
};

#endif