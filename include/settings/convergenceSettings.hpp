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

#ifndef SETTINGS__CONVERGENCE_SETTINGS_HPP
#define SETTINGS__CONVERGENCE_SETTINGS_HPP

#include "enums/convergence.hpp"
#include "setting.hpp"

/**
 * @class ConvSettings
 *
 * @brief stores all information about the optimizer
 *
 */
class ConvSettings
{
   private:
    Setting<double> _energyConv;
    Setting<double> _relEnergyConv;
    Setting<double> _absEnergyConv;

    Setting<double> _forceConv;
    Setting<double> _maxForceConv;
    Setting<double> _rmsForceConv;

    Setting<bool> _useEnergyConv;
    Setting<bool> _useForceConv;
    Setting<bool> _useMaxForceConv;
    Setting<bool> _useRMSForceConv;

    Setting<ConvStrategy> _energyConvStrategy;

   public:
    ConvSettings();

    /***************************
     * standard setter methods *
     ***************************/

    void setEnergyConv(double);
    void setRelEnergyConv(double);
    void setAbsEnergyConv(double);

    void setForceConv(double);
    void setMaxForceConv(double);
    void setRMSForceConv(double);

    void setUseEnergyConv(bool);
    void setUseForceConv(bool);
    void setUseMaxForceConv(bool);
    void setUseRMSForceConv(bool);

    void setEnergyConvStrategy(ConvStrategy);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] std::optional<double> getEnergyConv() const;
    [[nodiscard]] std::optional<double> getRelEnergyConv() const;
    [[nodiscard]] std::optional<double> getAbsEnergyConv() const;

    [[nodiscard]] std::optional<double> getForceConv() const;
    [[nodiscard]] std::optional<double> getMaxForceConv() const;
    [[nodiscard]] std::optional<double> getRMSForceConv() const;

    [[nodiscard]] bool getUseEnergyConv() const;
    [[nodiscard]] bool getUseForceConv() const;
    [[nodiscard]] bool getUseMaxForceConv() const;
    [[nodiscard]] bool getUseRMSForceConv() const;

    [[nodiscard]] ConvStrategy getEnConvStrategy() const;
};

#endif   // SETTINGS__CONVERGENCE_SETTINGS_HPP
