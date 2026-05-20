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

#ifndef _WATER_MODEL_SETUP_HPP_

#define _WATER_MODEL_SETUP_HPP_

#include <optional>

#include "interWater.hpp"
#include "typeAliases.hpp"
#include "waterModelSettings.hpp"

namespace setup
{
    struct RigidWaterGeometry
    {
        double dOH{0.0};
        double dHH{0.0};
    };

    void setupWaterModel(pq::Engine &);

    /**
     * @class WaterModelSetup
     *
     * @brief this class setups up the water model for the simulation
     *
     */
    class WaterModelSetup
    {
       private:
        pq::MDEngine &_engine;

        void makeInterWater();
        void checkTopologyFile();
        void checkMoldescriptorWaterCharge(const waterModel::InterWaterState &);
        void shakeSetupForRigidWater(const RigidWaterGeometry &geometry);
        [[nodiscard]] std::optional<RigidWaterGeometry> getRigidWaterGeometry(
            const settings::WaterIntraModel intraModel
        );

       public:
        explicit WaterModelSetup(pq::MDEngine &engine);

        void setup();
    };

}   // namespace setup

#endif   // _WATER_MODEL_SETUP_HPP_