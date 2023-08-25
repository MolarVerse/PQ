#ifndef _ENERGY_OUTPUT_HPP_

#define _ENERGY_OUTPUT_HPP_

#include "output.hpp"   // for Output

#include <cstddef>   // for size_t

namespace physicalData
{
    class PhysicalData;
}   // namespace physicalData

namespace output
{
    /**
     * @class EnergyOutput inherits from Output
     *
     * @brief Output file for energy, temperature and pressure
     *
     */
    class EnergyOutput : public Output
    {
      public:
        using Output::Output;

        void write(const size_t, const physicalData::PhysicalData &);
    };

}   // namespace output

#endif   // _ENERGY_OUTPUT_HPP_