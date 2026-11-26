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
#include "referenceSettings.hpp"

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
void ManostatSettings::finalize(ReferenceSettings& referenceSettings)
{
    if (!_fixedAxis.isSetByUser())
    {
        if (_manostatType.get() == ManostatType::NONE)
            _fixedAxis.setBySystem(FixedAxis::ALL);
        else
            _fixedAxis.setBySystem(FixedAxis::NONE);
    }

    _addReferenceFile(referenceSettings);
}

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
 * @brief adds the appropriate reference file for the current manostat type
 * to the given reference settings
 *
 * @param referenceSettings
 */
void ManostatSettings::_addReferenceFile(ReferenceSettings& referenceSettings)
{
    if (!_manostatType.isSetBySystem() && !_manostatType.isSetByUser())
        return;

    switch (_manostatType.get())
    {
        case ManostatType::BERENDSEN:
            referenceSettings.addReferenceFile(ReferenceFiles::BERENDSEN_FILE);
            break;
        case ManostatType::STOCHASTIC_RESCALING:
            referenceSettings.addReferenceFile(
                ReferenceFiles::STOCHASTIC_RESCALING_FILE
            );
            break;
        case ManostatType::NONE: break;
    }
}
