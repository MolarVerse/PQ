#ifndef _ENERGYOUTPUT_HPP_

#define _ENERGYOUTPUT_HPP_

#include "output.hpp"
#include "physicalData.hpp"

namespace output
{
    class EnergyOutput;
}

/**
 * @class EnergyOutput inherits from Output
 *
 * @brief Output file for energy
 *
 */
class output::EnergyOutput : public output::Output
{
  public:
    using output::Output::Output;

    void write(const size_t, const physicalData::PhysicalData &);
};

#endif   // _ENERGYOUTPUT_HPP_