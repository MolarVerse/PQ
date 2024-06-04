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

using namespace settings;
using namespace defaults;

/****************************
 *                          *
 * standard getters methods *
 *                          *
 ****************************/

/**
 * @brief Get the mol descriptor file name
 *
 * @return std::string
 */
std::string FileSettings::getMolDescriptorFileName()
{
    return _molDescriptorFileName;
}

/**
 * @brief Get the guff dat file name
 *
 * @return std::string
 */
std::string FileSettings::getGuffDatFileName() { return _guffDatFileName; }

/**
 * @brief Get the topology file name
 *
 * @return std::string
 */
std::string FileSettings::getTopologyFileName() { return _topologyFileName; }

/**
 * @brief Get the parameter file name
 *
 * @return std::string
 */
std::string FileSettings::getParameterFilename() { return _parameterFileName; }

/**
 * @brief Get the intra non bonded file name
 *
 * @return std::string
 */
std::string FileSettings::getIntraNonBondedFileName()
{
    return _intraNonBondedFileName;
}

/**
 * @brief Get the start file name
 *
 * @return std::string
 */
std::string FileSettings::getStartFileName() { return _startFileName; }

/**
 * @brief Get the ring polymer start file name
 *
 * @return std::string
 */
std::string FileSettings::getRingPolymerStartFileName()
{
    return _ringPolymerStartFileName;
}

/**
 * @brief Get the mShake file name
 *
 * @return std::string
 */
std::string FileSettings::getMShakeFileName() { return _mShakeFileName; }

/**
 * @brief Check if the topology file name is set
 *
 * @return bool
 */
bool FileSettings::isTopologyFileNameSet() { return _isTopologyFileNameSet; }

/**
 * @brief Check if the parameter file name is set
 *
 * @return bool
 */
bool FileSettings::isParameterFileNameSet() { return _isParameterFileNameSet; }

/**
 * @brief Check if the intra non bonded file name is set
 *
 * @return bool
 */
bool FileSettings::isIntraNonBondedFileNameSet()
{
    return _isIntraNonBondedFileNameSet;
}

/**
 * @brief Check if the ring polymer start file name is set
 *
 * @return bool
 */
bool FileSettings::isRingPolymerStartFileNameSet()
{
    return _isRingPolymerStartFileNameSet;
}

/**
 * @brief Check if the mShake file name is set
 *
 * @return bool
 */
bool FileSettings::isMShakeFileNameSet() { return _isMShakeFileNameSet; }

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the mol descriptor file name
 *
 * @param name
 */
void FileSettings::setMolDescriptorFileName(const std::string_view name)
{
    FileSettings::_molDescriptorFileName = name;
}

/**
 * @brief set the guff dat file name
 *
 * @param name
 */
void FileSettings::setGuffDatFileName(const std::string_view name)
{
    FileSettings::_guffDatFileName = name;
}

/**
 * @brief set the topology file name
 *
 * @param name
 */
void FileSettings::setTopologyFileName(const std::string_view name)
{
    FileSettings::_topologyFileName = name;
}

/**
 * @brief set the parameter file name
 *
 * @param name
 */
void FileSettings::setParameterFileName(const std::string_view name)
{
    FileSettings::_parameterFileName = name;
}

/**
 * @brief set the intra non bonded file name
 *
 * @param name
 */
void FileSettings::setIntraNonBondedFileName(const std::string_view name)
{
    FileSettings::_intraNonBondedFileName = name;
}

/**
 * @brief set the start file name
 *
 * @param name
 */
void FileSettings::setStartFileName(const std::string_view name)
{
    FileSettings::_startFileName = name;
}

/**
 * @brief set the ring polymer start file name
 *
 * @param name
 */
void FileSettings::setRingPolymerStartFileName(const std::string_view name)
{
    FileSettings::_ringPolymerStartFileName = name;
}

/**
 * @brief set the mShake file name
 *
 * @param name
 */
void FileSettings::setMShakeFileName(const std::string_view name)
{
    FileSettings::_mShakeFileName = name;
}

/**
 * @brief set the topology file name flag to is set
 *
 */
void FileSettings::setIsTopologyFileNameSet()
{
    FileSettings::_isTopologyFileNameSet = true;
}

/**
 * @brief set the parameter file name flag to is set
 *
 */
void FileSettings::setIsParameterFileNameSet()
{
    FileSettings::_isParameterFileNameSet = true;
}

/**
 * @brief set the intra non bonded file name flag to is set
 *
 */
void FileSettings::setIsIntraNonBondedFileNameSet()
{
    FileSettings::_isIntraNonBondedFileNameSet = true;
}

/**
 * @brief set the ring polymer start file name flag to is set
 *
 */
void FileSettings::setIsRingPolymerStartFileNameSet()
{
    FileSettings::_isRingPolymerStartFileNameSet = true;
}

/**
 * @brief set the mShake file name flag to is set
 *
 */
void FileSettings::setIsMShakeFileNameSet()
{
    FileSettings::_isMShakeFileNameSet = true;
}

/**
 * @brief set the topology file name flag to is not set
 *
 */
void FileSettings::unsetIsTopologyFileNameSet()
{
    FileSettings::_isTopologyFileNameSet = false;
}

/**
 * @brief set the parameter file name flag to is not set
 *
 */
void FileSettings::unsetIsParameterFileNameSet()
{
    FileSettings::_isParameterFileNameSet = false;
}

/**
 * @brief set the intra non bonded file name flag to is not set
 *
 */
void FileSettings::unsetIsIntraNonBondedFileNameSet()
{
    FileSettings::_isIntraNonBondedFileNameSet = false;
}

/**
 * @brief set the ring polymer start file name flag to is not set
 *
 */
void FileSettings::unsetIsRingPolymerStartFileNameSet()
{
    FileSettings::_isRingPolymerStartFileNameSet = false;
}

/**
 * @brief set the mShake file name flag to is not set
 *
 */
void FileSettings::unsetIsMShakeFileNameSet()
{
    FileSettings::_isMShakeFileNameSet = false;
}