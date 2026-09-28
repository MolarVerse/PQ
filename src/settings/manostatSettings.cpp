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

#include "manostatSettings.hpp"

namespace settings
{

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief sets the manostatType to enum in settings
     *
     * @param manostatType
     */
    void ManostatSettings::setManostatType(ManostatType manostatType)
    {
        _manostatType = manostatType;
        if (!_isFixedAxisSet)
        {
            using enum ManostatType;
            _fixedAxis =
                (_manostatType == NONE) ? FixedAxis::ALL : FixedAxis::NONE;
        }
    }

    /**
     * @brief sets the isotropy to enum in settings
     *
     * @param isotropy
     */
    void ManostatSettings::setIsotropy(Isotropy isotropy)
    {
        _isotropy = isotropy;
    }

    void ManostatSettings::setFixedAxis(FixedAxis fixedAxis)
    {
        _fixedAxis      = fixedAxis;
        _isFixedAxisSet = true;
    }

    void ManostatSettings::setIsFixedAxisSet(bool isSet)
    {
        _isFixedAxisSet = isSet;
    }

    /**
     * @brief sets the targetPressure to double in settings
     *
     * @param targetPressure
     */
    void ManostatSettings::setTargetPressure(double targetPressure)
    {
        _targetPressure = targetPressure;
    }

    /**
     * @brief sets the tauManostat to double in settings
     *
     * @param tauManostat
     */
    void ManostatSettings::setTauManostat(double tauManostat)
    {
        _tauManostat = tauManostat;
    }

    /**
     * @brief sets the compressibility to double in settings
     *
     * @param compressibility
     */
    void ManostatSettings::setCompressibility(double compressibility)
    {
        _compressibility = compressibility;
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get if manostat is Berendsen based
     *
     * @return bool
     */
    bool ManostatSettings::isBerendsenBased()
    {
        using enum ManostatType;

        return _manostatType == BERENDSEN ||
               _manostatType == STOCHASTIC_RESCALING;
    }

    /**
     * @brief get the manostatType
     *
     * @return ManostatType
     */
    ManostatType ManostatSettings::getManostatType() { return _manostatType; }

    /**
     * @brief get the isotropy
     *
     * @return Isotropy
     */
    Isotropy ManostatSettings::getIsotropy() { return _isotropy; }

    /**
     * @brief get the FixedAxis
     *
     * @return FixedAxis
     */
    FixedAxis ManostatSettings::getFixedAxis() { return _fixedAxis; }

    /**
     * @brief get whether FixedAxis was explicitly set
     *
     * @return bool
     */
    bool ManostatSettings::isFixedAxisSet() { return _isFixedAxisSet; }

    /**
     * @brief get the target pressure
     *
     * @return double
     */
    double ManostatSettings::getTargetPressure() { return _targetPressure; }

    /**
     * @brief get the tauManostat
     *
     * @return double
     */
    double ManostatSettings::getTauManostat() { return _tauManostat; }

    /**
     * @brief get the compressibility
     *
     * @return double
     */
    double ManostatSettings::getCompressibility() { return _compressibility; }

}   // namespace settings
