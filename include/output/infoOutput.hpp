#ifndef __INFOOUTPUT_HPP__

#define __INFOOUTPUT_HPP__

#include "output.hpp"
#include "physicalData.hpp"

#include <functional>

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
    void
    writeLeft(const double, const std::string_view &, const std::string_view &, std::ios_base &(std::ios_base &), const size_t);
    void
    writeRight(const double, const std::string_view &, const std::string_view &, std::ios_base &(std::ios_base &), const size_t);

  public:
    using Output::Output;

    void write(const double, const physicalData::PhysicalData &data);
};

#endif /* __INFOOUTPUT_HPP__ */