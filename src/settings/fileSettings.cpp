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

#include "fileSettings.hpp"

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief Check if the topology file name is set
 *
 * @return bool
 */
bool FileSettings::isTopologyFileNameSetByUser() const
{
    return _topologyFile.isSetByUser();
}

/**
 * @brief Check if the parameter file name is set
 *
 * @return bool
 */
bool FileSettings::isParameterFileNameSetByUser() const
{
    return _parameterFile.isSetByUser();
}

/**
 * @brief Check if the intra non bonded file name is set
 *
 * @return bool
 */
bool FileSettings::isIntraNonBondedFileNameSetByUser() const
{
    return _intraNonBondedFile.isSetByUser();
}

/**
 * @brief Check if the ring polymer start file name is set by the user
 *
 * @return bool
 */
bool FileSettings::isRingPolymerStartFileNameSetByUser() const
{
    return _rpmdStartFile.isSetByUser();
}

/**
 * @brief Check if the start file name is set by the user
 *
 * @return bool
 */
bool FileSettings::isStartFileNameSetByUser() const
{
    return _startFile.isSetByUser();
}

/**
 * @brief Check if the mShake file name is set by the user
 *
 * @return bool
 */
bool FileSettings::isMShakeFileNameSetByUser() const
{
    return _mShakeFile.isSetByUser();
}
