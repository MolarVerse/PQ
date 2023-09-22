#include "momentumOutput.hpp"

#include "physicalData.hpp"   // for PhysicalData

#include <format>    // for format
#include <fstream>   // for basic_ostream, ofstream

using output::MomentumOutput;

/**
 * @brief Write the momentum output
 *
 * @param step
 * @param data
 */
void MomentumOutput::write(const size_t step, const physicalData::PhysicalData &data)
{
    _fp << std::format("{:10d}\t", step);
    _fp << std::format("{:20.5e}\t", norm(data.getMomentum()));
    _fp << std::format("{:20.5e}\t", data.getMomentum()[0]);
    _fp << std::format("{:20.5e}\t", data.getMomentum()[1]);
    _fp << std::format("{:20.5e}\t", data.getMomentum()[2]);
    _fp << std::format("{:20.5e}\t", norm(data.getAngularMomentum()));
    _fp << std::format("{:20.5e}\t", data.getAngularMomentum()[0]);
    _fp << std::format("{:20.5e}\t", data.getAngularMomentum()[1]);
    _fp << std::format("{:20.5e}\t", data.getAngularMomentum()[2]);
}