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

#ifndef SETTINGS__RESET_KINETICS_SETTINGS_HPP
#define SETTINGS__RESET_KINETICS_SETTINGS_HPP

#include <cstddef>

#include "setting.hpp"

/**
 * @class ResetKineticsSettings
 *
 * @brief  class to store settings of reset kinetics
 *
 */
class ResetKineticsSettings
{
   private:
    Setting<size_t> _nScale;
    Setting<size_t> _fScale;
    Setting<size_t> _nReset;
    Setting<size_t> _fReset;
    Setting<size_t> _nResetAngular;
    Setting<size_t> _fResetAngular;
    Setting<size_t> _fResetForces;

   public:
    ResetKineticsSettings();

    void finalize();

    /***************************
     * standard setter methods *
     ***************************/

    void setNScaleByUser(size_t nScale);
    void setFScaleByUser(size_t fScale);
    void setNResetByUser(size_t nReset);
    void setFResetByUser(size_t fReset);
    void setNResetAngularByUser(size_t nResetAngular);
    void setFResetAngularByUser(size_t fResetAngular);
    void setFResetForcesByUser(size_t fResetForces);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] size_t getNScale() const;
    [[nodiscard]] size_t getFScale() const;
    [[nodiscard]] size_t getNReset() const;
    [[nodiscard]] size_t getFReset() const;
    [[nodiscard]] size_t getNResetAngular() const;
    [[nodiscard]] size_t getFResetAngular() const;
    [[nodiscard]] size_t getFResetForces() const;
};

#endif   // SETTINGS__RESET_KINETICS_SETTINGS_HPP
