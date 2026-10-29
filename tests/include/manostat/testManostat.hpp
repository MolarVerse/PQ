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

#ifndef _TEST_MANOSTAT_HPP_

#define _TEST_MANOSTAT_HPP_

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "manostat.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"

/**
 * @class TestManostat
 *
 * @brief Fixture for manostat tests.
 *
 */
class TestManostat : public ::testing::Test
{
   protected:
    std::unique_ptr<physicalData::PhysicalData> _data;
    std::unique_ptr<molsys::SimulationBox>      _box;
    std::unique_ptr<manostat::Manostat>         _manostat;
    std::unique_ptr<Settings>                   _settings;

    void SetUp() override
    {
        _settings = std::make_unique<Settings>();
        _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
        _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
        _settings->manostatSettings.setTargetPressure(1.0);
        _settings->manostatSettings.setTauManostat(1.0);
        _settings->manostatSettings.setCompressibility(0.03);
        _manostat =
            std::make_unique<manostat::Manostat>(_settings->manostatSettings);
        _data = std::make_unique<physicalData::PhysicalData>();

        _data->setVolume(2.0);
        _data->setVirial(diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0)));
        _data->setKineticEnergyMolecularVector(
            diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0))
        );
        _data->setKineticEnergyAtomicVector(
            diagonalMatrix(linalg::Vec3D(1.0, 1.0, 1.0))
        );

        _box = std::make_unique<molsys::SimulationBox>();
        molsys::OrthorhombicBox orthoBox{{cbrt(2.0), cbrt(2.0), cbrt(2.0)}};
        _box->setBox(orthoBox, false);
    }
};

#endif
