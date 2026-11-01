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
#include <unordered_map>

#include "defaults.hpp"
#include "enums/qm.hpp"
#include "setting.hpp"

struct TestQMSettings;   // forward declaration

#define SINGULARITY_  _SINGULARITY_
#define STATIC_BUILD_ _STATIC_BUILD_
#define SCRIPT_PATH_  _SCRIPT_PATH_

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
    constexpr static auto *_singularity = SINGULARITY_;
    constexpr static auto *_staticBuild = STATIC_BUILD_;
    std::string            _scriptPath  = SCRIPT_PATH_;

    Setting<QMMethod, "QM method">            _qmMethod;
    Setting<MaceModel, "MACE model">          _maceModel{MaceModel::MEDIUM};
    Setting<MaceModelType, "MACE model type"> _maceModelType{
        MaceModelType::MACE_MP
    };
    Setting<MaceMode, "MACE mode">     _maceMode{MaceMode::ACCURATE};
    Setting<SlakosType, "Slakos type"> _slakosType{SlakosType::NONE};
    Setting<XtbMethod, "XTB method">   _xtbMethod{XtbMethod::GFN2};

    Setting<std::string, "QM script">           _qmScript{""};
    Setting<std::string, "QM script full path"> _qmScriptFullPath{""};
    Setting<std::string, "MACE model path">     _maceModelPath{""};
    Setting<std::string, "Slakos path">         _slakosPath{""};
    Setting<std::string, "Fennol model path">   _fennolModelPath{""};

    Setting<bool, "Use dispersion correction"> _useDispersionCorrection{false};
    Setting<bool, "Remove net force">          _removeNetForce{false};
    Setting<bool, "Use third order DFTB">      _useThirdOrderDftb;
    Setting<bool, "Use GPU preprocessing">     _useGPUPreprocessing{true};

    Setting<std::unordered_map<std::string, double>, "Hubbard derivatives">
        _hubbardDerivs;

    Setting<double, "QM loop time limit"> _qmLoopTimeLimit{
        defaults::QM_LOOP_TIME_LIMIT_DEFAULT
    };

   public:
    void finalize();

    [[nodiscard]] bool isExternalQMRunner() const;

    /***************************
     * standard setter methods *
     ***************************/

    void setQMMethod(QMMethod method);

    void setMaceModel(MaceModel model);
    void setMaceModelType(MaceModelType model);
    void setMaceMode(MaceMode mode);
    void setMaceModelPath(const std::string &path);

    void setQMScript(const std::string &script);
    void setQMScriptFullPath(const std::string &script);

    void setSlakosType(SlakosType slakos, bool resolveBuiltInPath);
    void setSlakosType(SlakosType slakos);
    void setSlakosPath(const std::string &path);

    void setUseDispersionCorrection(bool use);
    void setRemoveNetForce(bool removeNetForce);
    void setUseThirdOrderDftb(bool use);
    void setHubbardDerivs(
        const std::unordered_map<std::string, double> &hubbardDerivs
    );

    void setXtbMethod(XtbMethod method);

    void setFennolModelPath(const std::string &path);
    void setUseGPUPreprocessing(bool use);

    void setQMLoopTimeLimit(double time);

    void setScriptPath(const std::string &scriptPath);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] QMMethod      getQMMethod() const;
    [[nodiscard]] bool          isQMMethodSet() const;
    [[nodiscard]] MaceModel     getMaceModel() const;
    [[nodiscard]] MaceModelType getMaceModelType() const;
    [[nodiscard]] MaceMode      getMaceMode() const;
    [[nodiscard]] std::string   getMaceModelPath() const;

    [[nodiscard]] std::string getQMScript() const;
    [[nodiscard]] std::string getQMScriptFullPath() const;

    [[nodiscard]] SlakosType  getSlakosType() const;
    [[nodiscard]] std::string getSlakosPath() const;

    [[nodiscard]] bool useDispersionCorr() const;
    [[nodiscard]] bool getRemoveNetForce() const;
    [[nodiscard]] bool useThirdOrderDftb() const;
    [[nodiscard]] bool isThirdOrderDftbSet() const;
    [[nodiscard]]
    std::unordered_map<std::string, double> getHubbardDerivs() const;
    [[nodiscard]] bool                      isHubbardDerivsSet() const;

    [[nodiscard]] XtbMethod getXtbMethod() const;

    [[nodiscard]] std::string getFennolModelPath() const;
    [[nodiscard]] bool        useGPUPreprocessing() const;

    [[nodiscard]] double getQMLoopTimeLimit() const;

    [[nodiscard]] std::string getScriptPath() const;

   private:
    void _setupQMScript();
    void _setupQMMethodAseDftbPlus();

    friend struct ::TestQMSettings;
};

#endif   // SETTINGS__QM_SETTINGS_HPP
