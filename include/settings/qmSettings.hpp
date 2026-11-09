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
class ReferenceSettings;

/**
 * @brief Class for managing MACE settings
 *
 */
class MaceSettings
{
   private:
    Setting<MaceModel, "MaceModel">         _maceModel{MaceModel::MEDIUM};
    Setting<MaceModelType, "MaceModelType"> _maceModelType{
        MaceModelType::MACE_MP
    };
    Setting<MaceMode, "MaceMode">         _maceMode{MaceMode::ACCURATE};
    Setting<std::string, "MaceModelPath"> _maceModelPath{""};

    // should stay private to control testing behavior and prevent accidental
    // modification
    bool                   _useForTesting{false};
    static constexpr auto *_testModelTypeStr = "MACECalculator";

   public:
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
 * @brief FennolSettings
 *
 * @details stores all information about the Fennol model and GPU preprocessing
 * settings
 *
 */
class FennolSettings
{
   private:
    Setting<std::string, "FennolModelPath"> _fennolModelPath{""};
    Setting<bool, "UseGPUPreprocessing">    _useGPUPreprocessing{true};

   public:
    void setFennolModelPathByUser(const std::string &path);
    void setUseGPUPreprocessingByUser(bool use);

    [[nodiscard]] const std::string &getFennolModelPath() const;
    [[nodiscard]] bool               useGPUPreprocessing() const;
};

/**
 * @brief DftbSettings
 *
 * @details stores all information about the DFTB model and related settings
 *
 */
class DftbSettings
{
   private:
    Setting<SlakosType, "SlakosType">  _slakosType{SlakosType::NONE};
    Setting<std::string, "SlakosPath"> _slakosPath{""};
    Setting<bool, "UseThirdOrderDftb"> _useThirdOrderDftb{false};

    Setting<std::unordered_map<std::string, double>, "HubbardDerivs">
        _hubbardDerivs;

   public:
    void finalize(ReferenceSettings &referenceSettings);

    void setSlakosTypeByUser(SlakosType slakos);
    void setSlakosPathByUser(const std::string &path);
    void setSlakosTypeByUser(SlakosType slakos, bool resolveBuiltInPath);
    void setUseThirdOrderDftbByUser(bool use);
    void setHubbardDerivsByUser(
        const std::unordered_map<std::string, double> &hubbardDerivs
    );

    [[nodiscard]] SlakosType         getSlakosType() const;
    [[nodiscard]] const std::string &getSlakosPath() const;
    [[nodiscard]] bool               useThirdOrderDftb() const;
    [[nodiscard]]
    std::optional<std::unordered_map<std::string, double>> getHubbardDerivs(
    ) const;
    [[nodiscard]] bool isHubbardDerivsSetByUser() const;
};

/**
 * @brief XtbSettings
 *
 * @details stores all information about the XTB method and related settings
 *
 */
class XtbSettings
{
   private:
    Setting<XtbMethod, "XtbMethod"> _xtbMethod{XtbMethod::GFN2};

   public:
    void                    setXtbMethodByUser(XtbMethod method);
    [[nodiscard]] XtbMethod getXtbMethod() const;
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
    Setting<QMMethod, "QMMethod"> _qmMethod{QMMethod::NONE};

    static inline std::string _qmScript;
    static inline std::string _qmScriptFullPath;

    static inline bool _useDispersionCorrection = false;
    static inline bool _removeNetForce          = false;

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

    static void setUseDispersionCorrection(bool use);
    static void setRemoveNetForce(bool removeNetForce);

    static void setQMLoopTimeLimit(double time);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] QMMethod getQMMethod() const;
    [[nodiscard]] bool     isQMMethodSetByUser() const;

    [[nodiscard]] static std::string getQMScript();
    [[nodiscard]] static std::string getQMScriptFullPath();

    [[nodiscard]] static bool useDispersionCorr();
    [[nodiscard]] static bool getRemoveNetForce();

    [[nodiscard]] static double getQMLoopTimeLimit();
};

#endif   // SETTINGS__QM_SETTINGS_HPP
