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

#include "defaults.hpp"
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
    Setting<double>       _relEnergyConv{defaults::REL_ENERGY_CONV_DEFAULT};
    Setting<EnergyPerMol> _absEnergyConv{defaults::ABS_ENERGY_CONV_DEFAULT};

    Setting<Force> _maxForceConv{defaults::MAX_FORCE_CONV_DEFAULT};
    Setting<Force> _rmsForceConv{defaults::RMS_FORCE_CONV_DEFAULT};

    Setting<bool> _useEnergyConv;
    Setting<bool> _useForceConv;
    Setting<bool> _useMaxForceConv;
    Setting<bool> _useRMSForceConv;

    Setting<ConvStrategy> _energyConvStrategy;

   public:
    ConvSettings();

   public:
    /***************************
     * standard setter methods *
     ***************************/

    void setRelEnergyConvByUser(double);
    void setAbsEnergyConvByUser(const EnergyPerMol&);

    void setMaxForceConvByUser(const Force&);
    void setRMSForceConvByUser(const Force&);

    void setUseEnergyConvByUser(bool);
    void setUseForceConvByUser(bool);
    void setUseMaxForceConvByUser(bool);
    void setUseRMSForceConvByUser(bool);

    void setEnergyConvStrategyByUser(ConvStrategy);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] double              getRelEnergyConv() const;
    [[nodiscard]] const EnergyPerMol& getAbsEnergyConv() const;

    [[nodiscard]] const Force& getMaxForceConv() const;
    [[nodiscard]] const Force& getRMSForceConv() const;

    [[nodiscard]] bool getUseEnergyConv() const;
    [[nodiscard]] bool getUseForceConv() const;
    [[nodiscard]] bool getUseMaxForceConv() const;
    [[nodiscard]] bool getUseRMSForceConv() const;

    [[nodiscard]] ConvStrategy getEnConvStrategy() const;

   private:
    [[nodiscard]] bool _anyConvergenceSetByUser() const;
};

#endif   // SETTINGS__CONVERGENCE_SETTINGS_HPP
