#ifndef _TRAJ_OUTPUT_HPP_

#define _TRAJ_OUTPUT_HPP_

#include "output.hpp"
#include "frame.hpp"

class TrajOutput : public Output
{
public:
    using Output::Output;

    void write(frameTools::Frame &);
};

#endif