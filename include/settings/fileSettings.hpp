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
        static inline std::string _molDescriptorFileName =
            defaults::_MOLDESCRIPTOR_FILENAME_DEFAULT_;

        static inline std::string _guffDatFileName =
            defaults::_GUFF_FILENAME_DEFAULT_;

        static inline std::string _topologyFileName;
        static inline std::string _parameterFileName;
        static inline std::string _intraNonBondedFileName;
        static inline std::string _startFileName;
        static inline std::string _ringPolymerStartFileName;
        static inline std::string _mShakeFileName;

        static bool inline _isTopologyFileNameSet         = false;
        static bool inline _isParameterFileNameSet        = false;
        static bool inline _isIntraNonBondedFileNameSet   = false;
        static bool inline _isRingPolymerStartFileNameSet = false;
        static bool inline _isMShakeFileNameSet           = false;

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

        [[nodiscard]] static bool isTopologyFileNameSet();
        [[nodiscard]] static bool isParameterFileNameSet();
        [[nodiscard]] static bool isIntraNonBondedFileNameSet();
        [[nodiscard]] static bool isRingPolymerStartFileNameSet();
        [[nodiscard]] static bool isMShakeFileNameSet();

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

        static void setIsTopologyFileNameSet();
        static void setIsParameterFileNameSet();
        static void setIsIntraNonBondedFileNameSet();
        static void setIsRingPolymerStartFileNameSet();
        static void setIsMShakeFileNameSet();

        static void unsetIsTopologyFileNameSet();
        static void unsetIsParameterFileNameSet();
        static void unsetIsIntraNonBondedFileNameSet();
        static void unsetIsRingPolymerStartFileNameSet();
        static void unsetIsMShakeFileNameSet();
    };

}   // namespace settings

#endif   // _FILE_SETTINGS_HPP_