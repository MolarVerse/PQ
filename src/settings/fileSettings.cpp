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

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief Get the mol descriptor file name
 *
 * @return std::string
 */
std::string FileSettings::getMolDescriptorFileName()
{
    return _molDescriptorFile;
}

/**
 * @brief Get the guff dat file name
 *
 * @return std::string
 */
std::string FileSettings::getGuffDatFileName() { return _guffDatFile; }

/**
 * @brief Get the topology file name
 *
 * @return std::string
 */
std::string FileSettings::getTopologyFileName() { return _topologyFile; }

/**
 * @brief Get the parameter file name
 *
 * @return std::string
 */
std::string FileSettings::getParameterFilename() { return _parameterFile; }

/**
 * @brief Get the intra non bonded file name
 *
 * @return std::string
 */
std::string FileSettings::getIntraNonBondedFileName()
{
    return _intraNonBondedFile;
}

/**
 * @brief Get the start file name
 *
 * @return std::string
 */
std::string FileSettings::getStartFileName() { return _startFile; }

/**
 * @brief Get the ring polymer start file name
 *
 * @return std::string
 */
std::string FileSettings::getRingPolymerStartFileName()
{
    return _rpmdStartFile;
}

/**
 * @brief Get the mShake file name
 *
 * @return std::string
 */
std::string FileSettings::getMShakeFileName() { return _mShakeFile; }

/**
 * @brief Get the DFTB setup file name
 *
 * @return std::string
 */
std::string FileSettings::getDFTBFileName() { return _dftbFile; }

/**
 * @brief Check if the topology file name is set
 *
 * @return bool
 */
bool FileSettings::isTopologyFileNameSet() { return _isTopologyFileSet; }

/**
 * @brief Check if the parameter file name is set
 *
 * @return bool
 */
bool FileSettings::isParameterFileNameSet() { return _isParameterFileSet; }

/**
 * @brief Check if the intra non bonded file name is set
 *
 * @return bool
 */
bool FileSettings::isIntraNonBondedFileNameSet()
{
    return _isIntraNonBondedFileSet;
}

/**
 * @brief Check if the ring polymer start file name is set
 *
 * @return bool
 */
bool FileSettings::isRingPolymerStartFileNameSet()
{
    return _isRPMDStartFileSet;
}

/**
 * @brief Check if the mShake file name is set
 *
 * @return bool
 */
bool FileSettings::isMShakeFileNameSet() { return _isMShakeFileSet; }

/**
 * @brief Check if the DFTB setup file name is set
 *
 * @return bool
 */
bool FileSettings::isDFTBFileNameSet() { return _isDFTBFileSet; }

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
    FileSettings::_molDescriptorFile = name;
}

/**
 * @brief set the guff dat file name
 *
 * @param name
 */
void FileSettings::setGuffDatFileName(const std::string_view name)
{
    FileSettings::_guffDatFile = name;
}

/**
 * @brief set the topology file name
 *
 * @param name
 */
void FileSettings::setTopologyFileName(const std::string_view name)
{
    FileSettings::_topologyFile = name;
}

/**
 * @brief set the parameter file name
 *
 * @param name
 */
void FileSettings::setParameterFileName(const std::string_view name)
{
    FileSettings::_parameterFile = name;
}

/**
 * @brief set the intra non bonded file name
 *
 * @param name
 */
void FileSettings::setIntraNonBondedFileName(const std::string_view name)
{
    FileSettings::_intraNonBondedFile = name;
}

/**
 * @brief set the start file name
 *
 * @param name
 */
void FileSettings::setStartFileName(const std::string_view name)
{
    FileSettings::_startFile = name;
}

/**
 * @brief set the ring polymer start file name
 *
 * @param name
 */
void FileSettings::setRingPolymerStartFileName(const std::string_view name)
{
    FileSettings::_rpmdStartFile = name;
}

/**
 * @brief set the mShake file name
 *
 * @param name
 */
void FileSettings::setMShakeFileName(const std::string_view name)
{
    FileSettings::_mShakeFile = name;
}

/**
 * @brief set the DFTB setup file name
 *
 * @param name
 */
void FileSettings::setDFTBFileName(const std::string_view name)
{
    FileSettings::_dftbFile = name;
}

/**
 * @brief set the topology file name flag to is set
 *
 */
void FileSettings::setIsTopologyFileNameSet()
{
    FileSettings::_isTopologyFileSet = true;
}

/**
 * @brief set the parameter file name flag to is set
 *
 */
void FileSettings::setIsParameterFileNameSet()
{
    FileSettings::_isParameterFileSet = true;
}

/**
 * @brief set the intra non bonded file name flag to is set
 *
 */
void FileSettings::setIsIntraNonBondedFileNameSet()
{
    FileSettings::_isIntraNonBondedFileSet = true;
}

/**
 * @brief set the ring polymer start file name flag to is set
 *
 */
void FileSettings::setIsRingPolymerStartFileNameSet()
{
    FileSettings::_isRPMDStartFileSet = true;
}

/**
 * @brief set the mShake file name flag to is set
 *
 */
void FileSettings::setIsMShakeFileNameSet()
{
    FileSettings::_isMShakeFileSet = true;
}

/**
 * @brief set the DFTB setup file name flag to is set
 *
 */
void FileSettings::setIsDFTBFileNameSet()
{
    FileSettings::_isDFTBFileSet = true;
}

/**
 * @brief set the topology file name flag to is not set
 *
 */
void FileSettings::unsetIsTopologyFileNameSet()
{
    FileSettings::_isTopologyFileSet = false;
}

/**
 * @brief set the parameter file name flag to is not set
 *
 */
void FileSettings::unsetIsParameterFileNameSet()
{
    FileSettings::_isParameterFileSet = false;
}

/**
 * @brief set the intra non bonded file name flag to is not set
 *
 */
void FileSettings::unsetIsIntraNonBondedFileNameSet()
{
    FileSettings::_isIntraNonBondedFileSet = false;
}

/**
 * @brief set the ring polymer start file name flag to is not set
 *
 */
void FileSettings::unsetIsRingPolymerStartFileNameSet()
{
    FileSettings::_isRPMDStartFileSet = false;
}

/**
 * @brief set the mShake file name flag to is not set
 *
 */
void FileSettings::unsetIsMShakeFileNameSet()
{
    FileSettings::_isMShakeFileSet = false;
}

/**
 * @brief set the DFTB setup file name flag to is not set
 *
 */
void FileSettings::unsetIsDFTBFileNameSet()
{
    FileSettings::_isDFTBFileSet = false;
}