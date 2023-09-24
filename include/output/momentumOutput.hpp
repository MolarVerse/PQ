#ifndef _MOMENTUM_OUTPUT_HPP_

#define _MOMENTUM_OUTPUT_HPP_

#include "output.hpp"   // for Output

#include <cstddef>   // for size_t

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace output
{
    /**
     * @class MomentumOutput inherits from Output
     *
     * @brief Output file for momentum and angular momentum vectors
     *
     */
    class MomentumOutput : public Output
    {
      public:
        using Output::Output;

        void write(const size_t step, const physicalData::PhysicalData &);
    };

}   // namespace output

#endif   // _MOMENTUM_OUTPUT_HPP_