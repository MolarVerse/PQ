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

#include "defaults.hpp"

/**
 * @brief Construct a new Manostat Settings object with default values
 *
 */
ManostatSettings::ManostatSettings()
    : _manostatType(ManostatType::NONE),
      _isotropy(Isotropy::ISOTROPIC),
      _tau(defaults::BERENDSEN_MANOSTAT_RELAX_TIME),
      _compressibility(defaults::COMPRESSIBILITY_WATER_DEFAULT)
{
}

/**
 * @brief finalizes the settings by setting the fixed axis if it was not
 * explicitly set
 *
 */
void ManostatSettings::finalize()
{
    if (!_fixedAxis.isSet())
    {
        if (_manostatType.get() == ManostatType::NONE)
            _fixedAxis.set(FixedAxis::ALL);
        else
            _fixedAxis.set(FixedAxis::NONE);
    }
}

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
    _manostatType.set(manostatType);
}

/**
 * @brief sets the isotropy to enum in settings
 *
 * @param isotropy
 */
void ManostatSettings::setIsotropy(Isotropy isotropy)
{
    _isotropy.set(isotropy);
}

void ManostatSettings::setFixedAxis(FixedAxis fixedAxis)
{
    _fixedAxis.set(fixedAxis);
}

/**
 * @brief sets the targetPressure to double in settings
 *
 * @param targetPressure
 */
void ManostatSettings::setTargetPressure(double targetPressure)
{
    _targetPressure.set(targetPressure);
}

/**
 * @brief sets the tau of the manostat to Time in settings
 *
 * @param tau
 */
void ManostatSettings::setTau(const Time& tau) { _tau.set(tau); }

/**
 * @brief sets the compressibility to double in settings
 *
 * @param compressibility
 */
void ManostatSettings::setCompressibility(double compressibility)
{
    _compressibility.set(compressibility);
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
bool ManostatSettings::isBerendsenBased() const
{
    using enum ManostatType;

    return _manostatType.get() == BERENDSEN ||
           _manostatType.get() == STOCHASTIC_RESCALING;
}

/**
 * @brief get the manostatType
 *
 * @return ManostatType
 */
ManostatType ManostatSettings::getManostatType() const
{
    return _manostatType.get();
}

/**
 * @brief get the isotropy
 *
 * @return Isotropy
 */
Isotropy ManostatSettings::getIsotropy() const { return _isotropy.get(); }

/**
 * @brief get the FixedAxis
 *
 * @return FixedAxis
 */
FixedAxis ManostatSettings::getFixedAxis() const { return _fixedAxis.get(); }

/**
 * @brief get whether FixedAxis was explicitly set
 *
 * @return bool
 */
bool ManostatSettings::isFixedAxisSet() const { return _fixedAxis.isSet(); }

/**
 * @brief get the target pressure
 *
 * @return double
 */
double ManostatSettings::getTargetPressure() const
{
    return _targetPressure.get();
}

/**
 * @brief get the tau of the manostat
 *
 * @return Time
 */
const Time& ManostatSettings::getTau() const { return _tau.get(); }

/**
 * @brief get the compressibility
 *
 * @return double
 */
double ManostatSettings::getCompressibility() const
{
    return _compressibility.get();
}
