/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "optOutput.hpp"

#include <format>   // for std::format

#include "optimizer.hpp"   // for Optimizer

using namespace output;
using std::format;

/**
 * @brief write the output file
 *
 * @param step
 * @param optimizer
 */
void OptOutput::write(const size_t step, const opt::Optimizer& optimizer)
{
    const auto conv = optimizer.getConvergence();

    const auto stepStr     = format("{:8d}\t", step);
    const auto absEnStr    = format("{:.8e}\t", conv.getAbsEnergy());
    const auto relEnStr    = format("{:.8e}\t", conv.getRelEnergy());
    const auto maxForceStr = format("{:.8e}\t", conv.getAbsMaxForce());
    const auto rmsForceStr = format("{:.8e}\t", conv.getAbsRMSForce());

    _fp << stepStr;
    _fp << absEnStr;
    _fp << relEnStr;
    _fp << maxForceStr;
    _fp << rmsForceStr;

    const auto convStrategy = conv.getEnergyConvStrategy();

    auto relEnConvStr    = std::format("{:3d}", 0);
    auto absEnConvStr    = std::format("{:3d}", 0);
    auto maxForceConvStr = std::format("{:3d}", 0);
    auto rmsForceConvStr = std::format("{:3d}", 0);

    const auto isEnergyConvEnabled   = conv.isEnergyConvEnabled();
    const auto isMaxForceConvEnabled = conv.isMaxForceConvEnabled();
    const auto isRMSForceConvEnabled = conv.isRMSForceConvEnabled();

    auto isRelEnergyEnabled = false;
    auto isAbsEnergyEnabled = false;

    if (isEnergyConvEnabled)
    {
        if (convStrategy == settings::ConvStrategy::RIGOROUS ||
            convStrategy == settings::ConvStrategy::LOOSE)
        {
            isRelEnergyEnabled = true;
            isAbsEnergyEnabled = true;
        }
        else if (convStrategy == settings::ConvStrategy::ABSOLUTE)
        {
            isAbsEnergyEnabled = true;
        }
        else if (convStrategy == settings::ConvStrategy::RELATIVE)
        {
            isRelEnergyEnabled = true;
        }
    }

    const auto isRelEnConv    = conv.isRelEnergyConv();
    const auto isAbsEnConv    = conv.isAbsEnergyConv();
    const auto isMaxForceConv = conv.isAbsMaxForceConv();
    const auto isRMSForceConv = conv.isAbsRMSForceConv();

    auto isRelEnConvInt    = isRelEnConv ? 1 : -1;
    auto isAbsEnConvInt    = isAbsEnConv ? 1 : -1;
    auto isMaxForceConvInt = isMaxForceConv ? 1 : -1;
    auto isRMSForceConvInt = isRMSForceConv ? 1 : -1;

    isRelEnConvInt    = isRelEnergyEnabled ? isRelEnConvInt : 0;
    isAbsEnConvInt    = isAbsEnergyEnabled ? isAbsEnConvInt : 0;
    isMaxForceConvInt = isMaxForceConvEnabled ? isMaxForceConvInt : 0;
    isRMSForceConvInt = isRMSForceConvEnabled ? isRMSForceConvInt : 0;

    _fp << std::format("{:3d}\t", isRelEnConvInt);
    _fp << std::format("{:3d}\t", isAbsEnConvInt);
    _fp << std::format("{:3d}\t", isMaxForceConvInt);
    _fp << std::format("{:3d}\t", isRMSForceConvInt);

    _fp << std::format("{:.8e}\t", conv.getRelEnergyConvThreshold());
    _fp << std::format("{:.8e}\t", conv.getAbsEnergyConvThreshold());
    _fp << std::format("{:.8e}\t", conv.getAbsMaxForceConvThreshold());
    _fp << std::format("{:.8e}\n", conv.getAbsRMSForceConvThreshold());

    _fp.flush();
}