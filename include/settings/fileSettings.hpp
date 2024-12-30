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

#ifndef _FILE_SETTINGS_HPP_

#define _FILE_SETTINGS_HPP_

#include <string>        // for string, allocator
#include <string_view>   // for string_view

#include "defaults.hpp"

namespace settings
{
    /**
     * @class FileSettings
     *
     * @brief static class to store settings of the files
     *
     */
    class FileSettings
    {
       private:
        // clang-format off
        static inline std::string _molDescriptorFile = defaults::_MOLDESCRIPTOR_FILE_DEFAULT_;
        // clang-format on

        static inline std::string _guffDatFile = defaults::_GUFF_FILE_DEFAULT_;

        static inline std::string _topologyFile;
        static inline std::string _parameterFile;
        static inline std::string _intraNonBondedFile;
        static inline std::string _startFile;
        static inline std::string _rpmdStartFile;
        static inline std::string _mShakeFile;
        static inline std::string _dftbFile = defaults::_DFTB_FILE_DEFAULT_;

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

        [[nodiscard]] static std::string getMolDescriptorFileName();
        [[nodiscard]] static std::string getGuffDatFileName();
        [[nodiscard]] static std::string getTopologyFileName();
        [[nodiscard]] static std::string getParameterFilename();
        [[nodiscard]] static std::string getIntraNonBondedFileName();
        [[nodiscard]] static std::string getStartFileName();
        [[nodiscard]] static std::string getRingPolymerStartFileName();
        [[nodiscard]] static std::string getMShakeFileName();
        [[nodiscard]] static std::string getDFTBFileName();

        [[nodiscard]] static bool isTopologyFileNameSet();
        [[nodiscard]] static bool isParameterFileNameSet();
        [[nodiscard]] static bool isIntraNonBondedFileNameSet();
        [[nodiscard]] static bool isRingPolymerStartFileNameSet();
        [[nodiscard]] static bool isMShakeFileNameSet();
        [[nodiscard]] static bool isDFTBFileNameSet();

        /********************
         * standard setters *
         ********************/

        static void setMolDescriptorFileName(const std::string_view name);
        static void setGuffDatFileName(const std::string_view name);
        static void setTopologyFileName(const std::string_view name);
        static void setParameterFileName(const std::string_view name);
        static void setIntraNonBondedFileName(const std::string_view name);
        static void setStartFileName(const std::string_view name);
        static void setRingPolymerStartFileName(const std::string_view name);
        static void setMShakeFileName(const std::string_view name);
        static void setDFTBFileName(const std::string_view name);

        static void setIsTopologyFileNameSet();
        static void setIsParameterFileNameSet();
        static void setIsIntraNonBondedFileNameSet();
        static void setIsRingPolymerStartFileNameSet();
        static void setIsMShakeFileNameSet();
        static void setIsDFTBFileNameSet();

        static void unsetIsTopologyFileNameSet();
        static void unsetIsParameterFileNameSet();
        static void unsetIsIntraNonBondedFileNameSet();
        static void unsetIsRingPolymerStartFileNameSet();
        static void unsetIsMShakeFileNameSet();
        static void unsetIsDFTBFileNameSet();
    };

}   // namespace settings

#endif   // _FILE_SETTINGS_HPP_