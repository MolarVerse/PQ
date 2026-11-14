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

    /********************
     * standard getters *
     ********************/

    [[nodiscard]] const std::string& getMolDescriptorFileName() const;
    [[nodiscard]] const std::string& getGuffDatFileName() const;

    [[nodiscard]] const std::string& getTopologyFileName() const;
    [[nodiscard]] const std::string& getParameterFileName() const;
    [[nodiscard]] const std::string& getIntraNonBondedFileName() const;

    [[nodiscard]] const std::string& getStartFileName() const;
    [[nodiscard]] const std::string& getRingPolymerStartFileName() const;

    [[nodiscard]] const std::string& getDFTBFileName() const;
    [[nodiscard]] const std::string& getTMFileName() const;

    [[nodiscard]] const std::string& getMShakeFileName() const;

    [[nodiscard]] const std::string& getOrcaFileName() const;
    [[nodiscard]] const std::string& getPointChargeFileName() const;
    [[nodiscard]] const std::string& getQMForcesTempFileName() const;
    [[nodiscard]] const std::string& getQMChargesTempFileName() const;
    [[nodiscard]] const std::string& getStressTensorTempFileName() const;

    [[nodiscard]] bool isTopologyFileNameSetByUser() const;
    [[nodiscard]] bool isParameterFileNameSetByUser() const;
    [[nodiscard]] bool isIntraNonBondedFileNameSetByUser() const;

    [[nodiscard]] bool isStartFileNameSetByUser() const;
    [[nodiscard]] bool isRingPolymerStartFileNameSetByUser() const;

    [[nodiscard]] bool isMShakeFileNameSetByUser() const;

    /********************
     * standard setters *
     ********************/

    void setMolDescriptorFileNameByUser(const std::string& name);
    void setGuffDatFileNameByUser(const std::string& name);

    void setTopologyFileNameByUser(const std::string& name);
    void setParameterFileNameByUser(const std::string& name);
    void setIntraNonBondedFileNameByUser(const std::string& name);

    void setStartFileNameByUser(const std::string& name);
    void setRingPolymerStartFileNameByUser(const std::string& name);

    void setDFTBFileNameByUser(const std::string& name);
    void setTMFileNameByUser(const std::string& name);

    void setMShakeFileNameByUser(const std::string& name);
    void setOrcaFileNameByUser(const std::string& name);
    void setPointChargeFileNameByUser(const std::string& name);
};

#endif   // SETTINGS__FILE_SETTINGS_HPP
