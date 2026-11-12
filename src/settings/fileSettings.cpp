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
 * @brief Get the mol descriptor file name
 *
 * @return std::string
 */
const std::string& FileSettings::getMolDescriptorFileName() const
{
    return _molDescriptorFile.get();
}

/**
 * @brief Get the guff dat file name
 *
 * @return std::string
 */
const std::string& FileSettings::getGuffDatFileName() const
{
    return _guffDatFile.get();
}

/**
 * @brief Get the topology file name
 *
 * @return std::string
 */
const std::string& FileSettings::getTopologyFileName() const
{
    return _topologyFile.get();
}

/**
 * @brief Get the parameter file name
 *
 * @return std::string
 */
const std::string& FileSettings::getParameterFileName() const
{
    return _parameterFile.get();
}

/**
 * @brief Get the intra non bonded file name
 *
 * @return std::string
 */
const std::string& FileSettings::getIntraNonBondedFileName() const
{
    return _intraNonBondedFile.get();
}

/**
 * @brief Get the start file name
 *
 * @return const std::string&
 */
const std::string& FileSettings::getStartFileName() const
{
    return _startFile.get();
}

/**
 * @brief Get the ring polymer start file name
 *
 * @return const std::string&
 */
const std::string& FileSettings::getRingPolymerStartFileName() const
{
    return _rpmdStartFile.get();
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
 * @brief Get the TM setup file name
 *
 * @return std::string
 */
std::string FileSettings::getTMFileName() { return _tmFile; }

/**
 * @brief Get the pointcharge file name
 *
 * @return std::string
 */
std::string FileSettings::getPointChargeFileName() { return _pointChargeFile; }

/*
 * @brief Get the QM forces temp file name
 *
 * @return std::string
 */
std::string FileSettings::getQMForcesTempFileName()
{
    return _qmForcesTempFile;
}

/**
 * @brief Get the QM charges temp file name
 *
 * @return std::string
 */
std::string FileSettings::getQMChargesTempFileName()
{
    return _qmChargesTempFile;
}

/**
 * @brief Get the stress tensor temp file name
 *
 * @return std::string
 */
std::string FileSettings::getStressTensorTempFileName()
{
    return _stressTensorTempFile;
}

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
void FileSettings::setMolDescriptorFileNameByUser(const std::string& name)
{
    _molDescriptorFile.setByUser(name);
}

/**
 * @brief set the guff dat file name
 *
 * @param name
 */
void FileSettings::setGuffDatFileNameByUser(const std::string& name)
{
    _guffDatFile.setByUser(name);
}

/**
 * @brief set the topology file name
 *
 * @param name
 */
void FileSettings::setTopologyFileNameByUser(const std::string& name)
{
    _topologyFile.setByUser(name);
}

/**
 * @brief set the parameter file name
 *
 * @param name
 */
void FileSettings::setParameterFileNameByUser(const std::string& name)
{
    _parameterFile.setByUser(name);
}

/**
 * @brief set the intra non bonded file name
 *
 * @param name
 */
void FileSettings::setIntraNonBondedFileNameByUser(const std::string& name)
{
    _intraNonBondedFile.setByUser(name);
}

/**
 * @brief set the start file name
 *
 * @param name
 */
void FileSettings::setStartFileNameByUser(const std::string& name)
{
    _startFile.setByUser(name);
}

/**
 * @brief set the ring polymer start file name
 *
 * @param name
 */
void FileSettings::setRingPolymerStartFileNameByUser(const std::string& name)
{
    _rpmdStartFile.setByUser(name);
}

/**
 * @brief set the mShake file name
 *
 * @param name
 */
void FileSettings::setMShakeFileName(const std::string_view& name)
{
    FileSettings::_mShakeFile = name;
}

/**
 * @brief set the DFTB setup file name
 *
 * @param name
 */
void FileSettings::setDFTBFileName(const std::string_view& name)
{
    FileSettings::_dftbFile = name;
}

/**
 * @brief set the TM setup file name
 *
 * @param name
 */
void FileSettings::setTMFileName(const std::string_view& name)
{
    FileSettings::_tmFile = name;
}

/**
 * @brief set the pointcharge file name
 *
 * @param name
 */
void FileSettings::setPointChargeFileName(const std::string_view& name)
{
    FileSettings::_pointChargeFile = name;
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
