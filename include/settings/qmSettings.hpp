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

#ifndef SETTINGS__QM_SETTINGS_HPP
#define SETTINGS__QM_SETTINGS_HPP

#include <string>
#include <string_view>
#include <unordered_map>

#include "defaults.hpp"
#include "enums/qm.hpp"
#include "setting.hpp"

std::string string(
    const std::unordered_map<std::string, double> &unordered_map
);

struct TestQMSetupAseUtil;

/**
 * @brief Class for managing MACE settings
 *
 */
class MaceSettings
{
   private:
    Setting<MaceModel>     _maceModel{MaceModel::MEDIUM};
    Setting<MaceModelType> _maceModelType{MaceModelType::MACE_MP};
    Setting<MaceMode>      _maceMode{MaceMode::ACCURATE};
    Setting<std::string>   _maceModelPath{""};

    // should stay private to control testing behavior and prevent accidental
    // modification
    bool                   _useForTesting{false};
    static constexpr auto *_testModelTypeStr = "MACECalculator";

   public:
    void finalize();

    void setMaceModelByUser(MaceModel model);
    void setMaceModelTypeByUser(MaceModelType model);
    void setMaceModeByUser(MaceMode mode);
    void setMaceModelPathByUser(const std::string &path);

    [[nodiscard]] MaceModel     getMaceModel() const;
    [[nodiscard]] std::string   getMaceModelName() const;
    [[nodiscard]] MaceModelType getMaceModelType() const;
    [[nodiscard]] std::string   getMaceModelTypeName() const;
    [[nodiscard]] MaceMode      getMaceMode() const;
    [[nodiscard]] std::string   getMaceModelPath() const;

    friend struct ::TestQMSetupAseUtil;
};

/**
 * @brief QMSettings
 *
 * @details stores all information about the external qm runner
 *
 */
class QMSettings
{
   private:
    Setting<QMMethod>        _qmMethod{QMMethod::NONE};
    static inline SlakosType _slakosType = SlakosType::NONE;
    static inline XtbMethod  _xtbMethod  = XtbMethod::GFN2;

    static inline std::string _qmScript;
    static inline std::string _qmScriptFullPath;
    static inline std::string _slakosPath;
    static inline std::string _fennolModelPath;

    static inline bool _useDispersionCorrection = false;
    static inline bool _removeNetForce          = false;
    static inline bool _useThirdOrderDftb       = false;
    static inline bool _isThirdOrderDftbSet     = false;
    static inline bool _isHubbardDerivsSet      = false;
    static inline bool _useGPUPreprocessing     = true;

    static inline std::unordered_map<std::string, double> _hubbardDerivs;

    static inline double _qmLoopTimeLimit =
        defaults::QM_LOOP_TIME_LIMIT_DEFAULT;

   public:
    void finalize();

    [[nodiscard]] bool isExternalQMRunner() const;

    /***************************
     * standard setter methods *
     ***************************/

    void setQMMethodByUser(QMMethod method);

    static void setQMScript(const std::string_view &script);
    static void setQMScriptFullPath(const std::string_view &script);

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

    static void setXtbMethod(XtbMethod method);

    static void setFennolModelPath(const std::string_view &path);
    static void setUseGPUPreprocessing(bool use);

    static void setQMLoopTimeLimit(double time);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] QMMethod getQMMethod() const;
    [[nodiscard]] bool     isQMMethodSetByUser() const;

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

   private:
    void _finalizeMace();
};

#endif   // SETTINGS__QM_SETTINGS_HPP
