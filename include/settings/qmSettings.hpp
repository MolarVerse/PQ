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
        ASEXTB,
        PYSCF,
        TURBOMOLE,
        MACE,
        FENNOL
    };

    /**
     * @class enum MaceModel
     *
     */
    enum class MaceModel : size_t
    {
        SMALL,
        MEDIUM,
        LARGE,
        SMALL0B,
        MEDIUM0B,
        SMALL0B2,
        MEDIUM0B2,
        LARGE0B2,
        MEDIUM0B3,
        MEDIUMMPA0,
        MEDIUMOMAT0,
        CUSTOM,
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
     * @class enum MaceMode
     *
     * @brief enum class for the MACE evaluation mode / kernel backend
     */
    enum class MaceMode : size_t
    {
        ACCURATE,
        FAST
    };

    /**
     * @class enum xtbMethod
     */
    enum class XtbMethod : size_t
    {
        GFN1,
        GFN2,
        IPEA1,
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
    std::string string(const MaceModel model);
    std::string string(const MaceModelType model);
    std::string string(const MaceMode mode);
    std::string string(const XtbMethod method);
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
        static inline MaceModel     _maceModel     = MaceModel::MEDIUM;
        static inline MaceModelType _maceModelType = MaceModelType::MACE_MP;
        static inline MaceMode      _maceMode      = MaceMode::ACCURATE;
        static inline SlakosType    _slakosType    = SlakosType::NONE;
        static inline XtbMethod     _xtbMethod     = XtbMethod::GFN2;

        static inline std::string _qmScript;
        static inline std::string _qmScriptFullPath;
        static inline std::string _maceModelPath;
        static inline std::string _slakosPath;
        static inline std::string _fennolModelPath;

        static inline bool _useDispersionCorrection = false;
        static inline bool _removeNetForce          = false;
        static inline bool _useThirdOrderDftb       = false;
        static inline bool _isThirdOrderDftbSet     = false;
        static inline bool _isHubbardDerivsSet      = false;
        static inline bool _useGPUPreprocessing     = true;

        static inline std::unordered_map<std::string, double> _hubbardDerivs;

        // clang-format off
        static inline double _qmLoopTimeLimit = defaults::QM_LOOP_TIME_LIMIT_DEFAULT;
        // clang-format on

       public:
        [[nodiscard]] static bool isExternalQMRunner();

        /***************************
         * standard setter methods *
         ***************************/

        static void setQMMethod(const std::string_view &method);
        static void setQMMethod(const QMMethod method);

        static void setMaceModel(const std::string_view &model);
        static void setMaceModel(const MaceModel model);
        static void setMaceModelType(const std::string_view &model);
        static void setMaceModelType(const MaceModelType model);
        static void setMaceMode(const std::string_view &mode);
        static void setMaceMode(const MaceMode mode);
        static void setMaceModelPath(const std::string_view &path);

        static void setQMScript(const std::string_view &script);
        static void setQMScriptFullPath(const std::string_view &script);

        static void setSlakosType(const std::string_view &slakos);
        static void setSlakosType(
            const SlakosType slakos,
            bool             resolveBuiltInPath
        );
        static void setSlakosType(const SlakosType slakos);
        static void setSlakosPath(const std::string_view &path);

        static void setUseDispersionCorrection(const bool use);
        static void setRemoveNetForce(const bool use);
        static void setUseThirdOrderDftb(const bool use);
        static void setIsThirdOrderDftbSet(const bool isThirdOrderDftbSet);
        static void setHubbardDerivs(
            const std::unordered_map<std::string, double> hubbardDerivs
        );
        static void setIsHubbardDerivsSet(const bool isHubbardDerivsSet);

        static void setXtbMethod(const std::string_view &method);
        static void setXtbMethod(const XtbMethod method);

        static void setFennolModelPath(const std::string_view &path);
        static void setUseGPUPreprocessing(const bool use);

        static void setQMLoopTimeLimit(const double time);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static QMMethod      getQMMethod();
        [[nodiscard]] static MaceModel     getMaceModel();
        [[nodiscard]] static MaceModelType getMaceModelType();
        [[nodiscard]] static MaceMode      getMaceMode();
        [[nodiscard]] static std::string   getMaceModelPath();

        [[nodiscard]] static std::string getQMScript();
        [[nodiscard]] static std::string getQMScriptFullPath();

        [[nodiscard]] static SlakosType  getSlakosType();
        [[nodiscard]] static std::string getSlakosPath();

        [[nodiscard]] static bool useDispersionCorr();
        [[nodiscard]] static bool getRemoveNetForce();
        [[nodiscard]] static bool useThirdOrderDftb();
        [[nodiscard]] static bool isThirdOrderDftbSet();
        [[nodiscard]] static std::unordered_map<std::string, double> getHubbardDerivs(
        );
        [[nodiscard]] static bool isHubbardDerivsSet();

        [[nodiscard]] static XtbMethod getXtbMethod();

        [[nodiscard]] static std::string getFennolModelPath();
        [[nodiscard]] static bool        useGPUPreprocessing();

        [[nodiscard]] static double getQMLoopTimeLimit();
    };
}   // namespace settings

#endif   // _QM_SETTINGS_HPP_
