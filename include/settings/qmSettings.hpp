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

#include <cstddef>         // for size_t
#include <string>          // for string
#include <string_view>     // for string_view
#include <unordered_map>   // for unordered_map

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
        ASEDFTBPLUS,
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

    /**
     * @class enum Slakos
     */
    enum class SlakosType : size_t
    {
        NONE,
        THREEOB,
        MATSCI,
        CUSTOM
    };

    std::string string(const QMMethod method);
    std::string string(const MaceModelSize model);
    std::string string(const MaceModelType model);
    std::string string(const SlakosType slakos);
    std::string string(
        const std::unordered_map<std::string, double> unordered_map
    );

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
        static inline MaceModelSize _maceModelSize = MaceModelSize::MEDIUM;
        static inline MaceModelType _maceModelType = MaceModelType::MACE_MP;
        static inline SlakosType    _slakosType    = SlakosType::NONE;

        static inline std::string _qmScript         = "";
        static inline std::string _qmScriptFullPath = "";
        static inline std::string _maceModelPath    = "";
        static inline std::string _slakosPath       = "";

        static inline bool _useDispersionCorrection = false;
        static inline bool _useThirdOrderDftb       = false;
        static inline bool _isThirdOrderDftbSet     = false;
        static inline bool _isHubbardDerivsSet      = false;

        static inline std::unordered_map<std::string, double> _hubbardDerivs;

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

        static void setSlakosType(const std::string_view &slakos);
        static void setSlakosType(const SlakosType slakos);
        static void setSlakosPath(const std::string_view &path);

        static void setUseDispersionCorrection(const bool use);
        static void setUseThirdOrderDftb(const bool use);
        static void setIsThirdOrderDftbSet(const bool isThirdOrderDftbSet);
        static void setHubbardDerivs(
            const std::unordered_map<std::string, double> hubbardDerivs
        );
        static void setIsHubbardDerivsSet(const bool isHubbardDerivsSet);

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

        [[nodiscard]] static SlakosType  getSlakosType();
        [[nodiscard]] static std::string getSlakosPath();

        [[nodiscard]] static bool useDispersionCorr();
        [[nodiscard]] static bool useThirdOrderDftb();
        [[nodiscard]] static bool isThirdOrderDftbSet();
        [[nodiscard]] static std::unordered_map<std::string, double> getHubbardDerivs(
        );
        [[nodiscard]] static bool isHubbardDerivsSet();

        [[nodiscard]] static double getQMLoopTimeLimit();
    };
}   // namespace settings

#endif   // _QM_SETTINGS_HPP_