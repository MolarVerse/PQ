#ifndef _ENERGY_OUTPUT_HPP_

#define _ENERGY_OUTPUT_HPP_

#include "output.hpp"
#include "physicalData.hpp"

namespace output
{
    class EnergyOutput;
}

/**
 * @class EnergyOutput inherits from Output
 *
 * @brief Output file for energy, temperature and pressure
 *
 */
class output::EnergyOutput : public output::Output
{
  public:
    using output::Output::Output;

    void write(const size_t, const physicalData::PhysicalData &);
};

#endif   // _ENERGY_OUTPUT_HPP_