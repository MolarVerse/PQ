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
#include <string_view>

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

    Setting<std::string, "TopologyFile">  _topologyFile;
    Setting<std::string, "ParameterFile"> _parameterFile;

    static inline std::string _intraNonBondedFile;
    static inline std::string _startFile;
    static inline std::string _rpmdStartFile;
    static inline std::string _mShakeFile;
    static inline std::string _dftbFile = defaults::DFTB_FILE_DEFAULT;
    static inline std::string _tmFile   = defaults::TM_FILE_DEFAULT;
    static inline std::string _orcaFile = defaults::ORCA_FILE_DEFAULT;
    static inline std::string _pointChargeFile =
        defaults::POINTCHARGE_FILE_DEFAULT;

    static inline std::string _qmForcesTempFile =
        defaults::QM_FORCES_TEMP_FILE_DEFAULT;
    static inline std::string _qmChargesTempFile =
        defaults::QM_CHARGES_TEMP_FILE_DEFAULT;
    static inline std::string _stressTensorTempFile =
        defaults::STRESS_TENSOR_TEMP_FILE_DEFAULT;

    static bool inline _isTopologyFileSet       = false;
    static bool inline _isParameterFileSet      = false;
    static bool inline _isIntraNonBondedFileSet = false;
    static bool inline _isRPMDStartFileSet      = false;
    static bool inline _isMShakeFileSet         = false;
    static bool inline _isDFTBFileSet           = false;

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

    [[nodiscard]] static std::string getIntraNonBondedFileName();
    [[nodiscard]] static std::string getStartFileName();
    [[nodiscard]] static std::string getRingPolymerStartFileName();
    [[nodiscard]] static std::string getMShakeFileName();
    [[nodiscard]] static std::string getDFTBFileName();
    [[nodiscard]] static std::string getTMFileName();
    [[nodiscard]] static std::string getOrcaFileName();
    [[nodiscard]] static std::string getPointChargeFileName();

    [[nodiscard]] static std::string getQMForcesTempFileName();
    [[nodiscard]] static std::string getQMChargesTempFileName();
    [[nodiscard]] static std::string getStressTensorTempFileName();

    [[nodiscard]] bool isTopologyFileNameSetByUser() const;
    [[nodiscard]] bool isParameterFileNameSetByUser() const;

    [[nodiscard]] static bool isIntraNonBondedFileNameSet();
    [[nodiscard]] static bool isRingPolymerStartFileNameSet();
    [[nodiscard]] static bool isMShakeFileNameSet();
    [[nodiscard]] static bool isDFTBFileNameSet();

    /********************
     * standard setters *
     ********************/

    void setMolDescriptorFileNameByUser(const std::string& name);
    void setGuffDatFileNameByUser(const std::string& name);

    void setTopologyFileNameByUser(const std::string& name);
    void setParameterFileNameByUser(const std::string& name);

    static void setIntraNonBondedFileName(const std::string_view& name);
    static void setStartFileName(const std::string_view& name);
    static void setRingPolymerStartFileName(const std::string_view& name);
    static void setMShakeFileName(const std::string_view& name);
    static void setDFTBFileName(const std::string_view& name);
    static void setTMFileName(const std::string_view& name);
    static void setOrcaFileName(const std::string_view& name);
    static void setPointChargeFileName(const std::string_view& name);

    static void setIsIntraNonBondedFileNameSet();
    static void setIsRingPolymerStartFileNameSet();
    static void setIsMShakeFileNameSet();
    static void setIsDFTBFileNameSet();

    static void unsetIsIntraNonBondedFileNameSet();
    static void unsetIsRingPolymerStartFileNameSet();
    static void unsetIsMShakeFileNameSet();
    static void unsetIsDFTBFileNameSet();
};

#endif   // SETTINGS__FILE_SETTINGS_HPP
