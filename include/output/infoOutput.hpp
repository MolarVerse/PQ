#ifndef __INFOOUTPUT_HPP__

#define __INFOOUTPUT_HPP__

#include "output.hpp"
#include "physicalData.hpp"

/**
 * @class InfoOutput inherits from Output
 *
 * @brief Output file for info file
 *
 */
class InfoOutput : public Output
{
public:
    using Output::Output;

    void write(const double, const PhysicalData &data);
};

#endif /* __INFOOUTPUT_HPP__ */