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

#ifndef _RESET_KINETICS_HPP_

#define _RESET_KINETICS_HPP_

#include <cstddef>   // for size_t

#include "timer.hpp"   // for Timer
#include "typeAliases.hpp"
#include "vector3d.hpp"   // for Vec3D

namespace resetKinetics
{
    /**
     * @class ResetKinetics
     *
     * @brief base class for the reset of the kinetics - represents also class
     * for no reset
     *
     */
    class ResetKinetics : public timings::Timer
    {
       protected:
        size_t _nStepsTemperatureReset;
        size_t _frequencyTemperatureReset;
        size_t _nStepsMomentumReset;
        size_t _frequencyMomentumReset;
        size_t _nStepsAngularReset;
        size_t _frequencyAngularReset;

        double    _temperature = 0.0;
        pq::Vec3D _momentum;
        pq::Vec3D _angularMomentum;

       public:
        ResetKinetics() = default;
        ResetKinetics(
            const size_t nStepsTemperatureReset,
            const size_t frequencyTemperatureReset,
            const size_t nStepsMomentumReset,
            const size_t frequencyMomentumReset,
            const size_t nStepsAngularReset,
            const size_t frequencyAngularReset
        );

        void reset(const size_t step, pq::PhysicalData &, pq::SimBox &);
        void resetTemperature(pq::SimBox &);
        void resetMomentum(pq::SimBox &);
        void resetAngularMomentum(pq::SimBox &);

        /********************
         * standard setters *
         *******************/

        void setTemperature(const double temperature);
        void setMomentum(const pq::Vec3D &momentum);
        void setAngularMomentum(const pq::Vec3D &angularMomentum);

        /********************
         * standard getters *
         *******************/

        [[nodiscard]] size_t getNStepsTemperatureReset() const;
        [[nodiscard]] size_t getFrequencyTemperatureReset() const;
        [[nodiscard]] size_t getNStepsMomentumReset() const;
        [[nodiscard]] size_t getFrequencyMomentumReset() const;
    };

}   // namespace resetKinetics

#endif   // _RESET_KINETICS_HPP_