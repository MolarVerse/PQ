#ifndef _TRAJ_OUTPUT_HPP_

#define _TRAJ_OUTPUT_HPP_

#include "frame.hpp"
#include "output.hpp"

class TrajOutput : public output::Output
{
  public:
    using output::Output::Output;

    void write(frameTools::Frame &);
};

#endif