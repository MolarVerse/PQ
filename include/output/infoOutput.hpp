#ifndef __INFO_OUTPUT_HPP__

#define __INFO_OUTPUT_HPP__

#include "output.hpp"   // for Output

#include <cstddef>       // for size_t
#include <ios>           // for ios_base
#include <string_view>   // for string_view

namespace physicalData
{
    class PhysicalData;
}   // namespace physicalData

namespace output
{

    /**
     * @class InfoOutput inherits from Output
     *
     * @brief Output file for info file
     *
     */
    class InfoOutput : public Output
    {
      private:
        void writeHeader();
        void writeLeft(
            const double, const std::string_view &, const std::string_view &, std::ios_base &(std::ios_base &), const size_t);
        void writeRight(
            const double, const std::string_view &, const std::string_view &, std::ios_base &(std::ios_base &), const size_t);

      public:
        using Output::Output;

        void write(const double, const physicalData::PhysicalData &data);
    };

}   // namespace output

#endif /* __INFO_OUTPUT_HPP__ */