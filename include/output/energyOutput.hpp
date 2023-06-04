#ifndef _ENERGYOUTPUT_HPP_

#define _ENERGYOUTPUT_HPP_

#include "output.hpp"
#include "physicalData.hpp"

/**
 * @class EnergyOutput inherits from Output
 *
 * @brief Output file for energy
 *
 */
class EnergyOutput : public Output
{
public:
    using Output::Output;

    void write(const size_t, const PhysicalData &);
};

#endif // _ENERGYOUTPUT_HPP_