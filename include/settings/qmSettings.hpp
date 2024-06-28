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

#ifndef _QM_SETTINGS_HPP_

#define _QM_SETTINGS_HPP_

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view

#include "defaults.hpp"   // for _QM_LOOP_TIME_LIMIT_DEFAULT_

namespace settings
{
    /**
     * @class enum QMMethod
     *
     */
    enum class QMMethod : size_t
    {
        NONE,
        DFTBPLUS,
        PYSCF,
        TURBOMOLE,
        MACE
    };

    /**
     * @class enum MaceModelSize
     *
     */
    enum class MaceModelSize : size_t
    {
        LARGE,
        MEDIUM,
        SMALL
    };

    /**
     * @class enum MaceModelType
     */
    enum class MaceModelType : size_t
    {
        MACE_MP,
        MACE_OFF,
        MACE_ANICC
    };

    std::string string(const QMMethod method);
    std::string string(const MaceModelSize model);
    std::string string(const MaceModelType model);

    /**
     * @class QMSettings
     *
     * @brief stores all information about the external qm runner
     *
     */
    class QMSettings
    {
       private:
        static inline QMMethod      _qmMethod      = QMMethod::NONE;
        static inline MaceModelSize _maceModelSize = MaceModelSize::LARGE;
        static inline MaceModelType _maceModelType = MaceModelType::MACE_MP;

        static inline std::string _qmScript         = "";
        static inline std::string _qmScriptFullPath = "";
        static inline std::string _maceModelPath    = "";

        static inline bool _useDispersionCorrection = false;

        // clang-format off
        static inline double _qmLoopTimeLimit = defaults::_QM_LOOP_TIME_LIMIT_DEFAULT_;
        // clang-format on

       public:
        [[nodiscard]] static bool isExternalQMRunner();

        /***************************
         * standard setter methods *
         ***************************/

        static void setQMMethod(const std::string_view &method);
        static void setQMMethod(const QMMethod method);

        static void setMaceModelSize(const std::string_view &model);
        static void setMaceModelSize(const MaceModelSize model);
        static void setMaceModelType(const std::string_view &model);
        static void setMaceModelType(const MaceModelType model);
        static void setMaceModelPath(const std::string_view &path);

        static void setQMScript(const std::string_view &script);
        static void setQMScriptFullPath(const std::string_view &script);

        static void setUseDispersionCorrection(const bool use);

        static void setQMLoopTimeLimit(const double time);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static QMMethod      getQMMethod();
        [[nodiscard]] static MaceModelSize getMaceModelSize();
        [[nodiscard]] static MaceModelType getMaceModelType();
        [[nodiscard]] static std::string   getMaceModelPath();

        [[nodiscard]] static std::string getQMScript();
        [[nodiscard]] static std::string getQMScriptFullPath();

        [[nodiscard]] static bool useDispersionCorrection();

        [[nodiscard]] static double getQMLoopTimeLimit();
    };
}   // namespace settings

#endif   // _QM_SETTINGS_HPP_