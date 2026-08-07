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

#ifndef _ADAM_HPP_

#define _ADAM_HPP_

#include <vector>   // for vector

#include "optimizer.hpp"

namespace opt
{
    /**
     * @class Adam
     *
     */
    class Adam : public Optimizer
    {
       private:
        constexpr static size_t _maxHistoryLength = 2;

        double _beta1 = 0.9;
        double _beta2 = 0.999;

        std::vector<linearAlgebra::Vec3D> _momentum1;
        std::vector<linearAlgebra::Vec3D> _momentum2;

       public:
        explicit Adam(const size_t nEpochs, const size_t nAtoms);
        explicit Adam(const size_t, const double, const double, const size_t);

        Adam()        = default;
        ~Adam() final = default;

        [[nodiscard]] pq::SharedOptimizer clone() const final;
        [[nodiscard]] size_t              maxHistoryLength() const final;

        void update(const double learningRate, const size_t step) final;
    };
}   // namespace opt

#endif   // _ADAM_HPP_