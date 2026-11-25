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

#ifndef SETTINGS__FILE_SETTINGS_HPP
#define SETTINGS__FILE_SETTINGS_HPP

#include <string>

#include "defaults.hpp"
#include "setting.hpp"

/**
 * @class FileSettings
 *
 * @brief static class to store settings of the files
 *
 */
class FileSettings
{
   private:
    Setting<std::string, "MolDescriptorFile"> _molDescriptorFile{
        defaults::MOLDESCRIPTOR_FILE_DEFAULT
    };

    Setting<std::string, "GuffDatFile"> _guffDatFile{
        defaults::GUFF_FILE_DEFAULT
    };

    Setting<std::string, "TopologyFile">       _topologyFile;
    Setting<std::string, "ParameterFile">      _parameterFile;
    Setting<std::string, "IntraNonBondedFile"> _intraNonBondedFile;

    Setting<std::string, "StartFile">     _startFile;
    Setting<std::string, "RPMDStartFile"> _rpmdStartFile;

    Setting<std::string, "DFTBFile"> _dftbFile{defaults::DFTB_FILE_DEFAULT};
    Setting<std::string, "TMFile">   _tmFile{defaults::TM_FILE_DEFAULT};

    Setting<std::string, "OrcaFile"> _orcaFile{defaults::ORCA_FILE_DEFAULT};
    Setting<std::string, "PointChargeFile"> _pointChargeFile{
        defaults::POINTCHARGE_FILE_DEFAULT
    };
    Setting<std::string, "QMForcesTempFile"> _qmForcesTempFile{
        defaults::QM_FORCES_TEMP_FILE_DEFAULT
    };
    Setting<std::string, "QMChargesTempFile"> _qmChargesTempFile{
        defaults::QM_CHARGES_TEMP_FILE_DEFAULT
    };
    Setting<std::string, "StressTensorTempFile"> _stressTensorTempFile{
        defaults::STRESS_TENSOR_TEMP_FILE_DEFAULT
    };

    Setting<std::string, "MShakeFile"> _mShakeFile;

   public:
    FileSettings()  = default;
    ~FileSettings() = default;

    SETTING_GETTER_AND_SETTER_BY_USER(_molDescriptorFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_guffDatFile);

    SETTING_GETTER_AND_SETTER_BY_USER(_topologyFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_parameterFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_intraNonBondedFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_mShakeFile);

    SETTING_GETTER_AND_SETTER_BY_USER(_startFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_rpmdStartFile);

    SETTING_GETTER_AND_SETTER_BY_USER(_dftbFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_tmFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_orcaFile);

    SETTING_GETTER_AND_SETTER_BY_USER(_pointChargeFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_qmForcesTempFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_qmChargesTempFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_stressTensorTempFile);

    [[nodiscard]] bool isTopologyFileNameSetByUser() const;
    [[nodiscard]] bool isParameterFileNameSetByUser() const;
    [[nodiscard]] bool isIntraNonBondedFileNameSetByUser() const;

    [[nodiscard]] bool isStartFileNameSetByUser() const;
    [[nodiscard]] bool isRingPolymerStartFileNameSetByUser() const;

    [[nodiscard]] bool isMShakeFileNameSetByUser() const;
};

#endif   // SETTINGS__FILE_SETTINGS_HPP
