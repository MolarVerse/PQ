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

#ifndef SETUP__QM_SETUP_HPP
#define SETUP__QM_SETUP_HPP

#include <mstd/types.hpp>

#include "qmCapableEngine.hpp"
namespace engine
{
    class Engine;       // forward declaration
    class QMMDEngine;   // forward declaration
}   // namespace engine

class ReferenceSettings;
struct TestQMSetupUtils;

namespace setup
{
    void setupQM(engine::Engine &, ReferenceSettings &references);

    /**
     * @class QMSetup
     *
     * @brief Setup QM
     *
     */
    class QMSetup
    {
       private:
        engine::QMCapableEngine       &_qmCapableEngine;
        mstd::ConstRef<QMSettings>     _qmSettings;
        mstd::ConstRef<MaceSettings>   _maceSettings;
        mstd::ConstRef<FennolSettings> _fennolSettings;
        mstd::ConstRef<DftbSettings>   _dftbSettings;
        mstd::ConstRef<XtbSettings>    _xtbSettings;

       public:
        explicit QMSetup(
            engine::QMCapableEngine &qmCapableEngine,
            const QMSettings        &qmSettings,
            const MaceSettings      &maceSettings,
            const FennolSettings    &fennolSettings,
            const DftbSettings      &dftbSettings,
            const XtbSettings       &xtbSettings
        );

        void        setup(ReferenceSettings &references);
        void        setupQMMethod();
        void        setupQMScript() const;
        static void setupCoulombRadiusCutOff();
        void        setupWriteInfo() const;

       private:
        void _setupQMMethodAseXtb(ReferenceSettings &references) const;

        friend struct ::TestQMSetupUtils;
    };

}   // namespace setup

#endif   // SETUP__QM_SETUP_HPP
