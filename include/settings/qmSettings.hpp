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

#include <cstdint>
#include <string>          // for string
#include <string_view>     // for string_view
#include <unordered_map>   // for unordered_map

#include "defaults.hpp"   // for _QM_LOOP_TIME_LIMIT_DEFAULT_

namespace settings
{
    /**
     * @brief enum QMMethod
     *
     */
    enum class QMMethod : std::uint8_t
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
     * @brief enum MaceModel
     *
     */
    enum class MaceModel : std::uint8_t
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
     * @brief enum MaceModelType
     */
    enum class MaceModelType : std::uint8_t
    {
        MACE_MP,
        MACE_OFF,
        MACE_ANICC
    };

    /**
     * @brief enum MaceMode
     *
     * @details enum class for the MACE evaluation mode / kernel backend
     */
    enum class MaceMode : std::uint8_t
    {
        ACCURATE,
        FAST
    };

    /**
     * @brief enum XtbMethod
     */
    enum class XtbMethod : std::uint8_t
    {
        GFN1,
        GFN2,
        IPEA1,
    };

    /**
     * @brief enum SlakosType
     */
    enum class SlakosType : std::uint8_t
    {
        NONE,
        THREEOB,
        MATSCI,
        CUSTOM
    };

    std::string string(QMMethod method);
    std::string string(MaceModel model);
    std::string string(MaceModelType model);
    std::string string(MaceMode mode);
    std::string string(XtbMethod method);
    std::string string(SlakosType slakos);
    std::string string(
        const std::unordered_map<std::string, double> &unordered_map
    );

    /**
     * @brief QMSettings
     *
     * @details stores all information about the external qm runner
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
        static void setQMMethod(QMMethod method);

        static void setMaceModel(const std::string_view &model);
        static void setMaceModel(MaceModel model);
        static void setMaceModelType(const std::string_view &model);
        static void setMaceModelType(MaceModelType model);
        static void setMaceMode(const std::string_view &mode);
        static void setMaceMode(MaceMode mode);
        static void setMaceModelPath(const std::string_view &path);

        static void setQMScript(const std::string_view &script);
        static void setQMScriptFullPath(const std::string_view &script);

        static void setSlakosType(const std::string_view &slakos);
        static void setSlakosType(SlakosType slakos, bool resolveBuiltInPath);
        static void setSlakosType(SlakosType slakos);
        static void setSlakosPath(const std::string_view &path);

        static void setUseDispersionCorrection(bool use);
        static void setRemoveNetForce(bool removeNetForce);
        static void setUseThirdOrderDftb(bool use);
        static void setIsThirdOrderDftbSet(bool isThirdOrderDftbSet);
        static void setHubbardDerivs(
            const std::unordered_map<std::string, double> &hubbardDerivs
        );
        static void setIsHubbardDerivsSet(bool isHubbardDerivsSet);

        static void setXtbMethod(const std::string_view &method);
        static void setXtbMethod(XtbMethod method);

        static void setFennolModelPath(const std::string_view &path);
        static void setUseGPUPreprocessing(bool use);

        static void setQMLoopTimeLimit(double time);

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
