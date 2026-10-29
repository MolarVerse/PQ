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

#ifndef _STOCHASTIC_RESCALING_MANOSTAT_HPP_

#define _STOCHASTIC_RESCALING_MANOSTAT_HPP_

#include "enums/manostat.hpp"
#include "manostat.hpp"
#include "randomNumberGenerator.hpp"

namespace manostat
{
    /**
     * @brief StochasticRescalingManostat inherits from Manostat
     *
     * _(https://doi.org/10.1063/5.0020514)
     *
     */
    class StochasticRescalingManostat : public Manostat
    {
       private:
        rng::RandomNumberGenerator _randomNumberGenerator;

        double    _tau;
        double    _compressibility;
        FixedAxis _fixedAxis;

       public:
        using Manostat::Manostat;

        void applyManostat(
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData
        ) override;

        [[nodiscard]] virtual linalg::tensor3D calculateMu(double);

        [[nodiscard]] double getTau() const;
        [[nodiscard]] double getCompressibility() const;

       protected:
        [[nodiscard]]
        bool _hasFixedAxis(size_t index) const;

        [[nodiscard]]
        rng::RandomNumberGenerator &_getRNG();
    };

    /**
     * @brief SemiIsotropicStochasticRescalingManostat inherits from Manostat
     *
     * _(https://doi.org/10.1063/5.0020514)
     *
     */
    class SemiIsotropicStochasticRescalingManostat
        : public StochasticRescalingManostat
    {
       private:
        Isotropy _isotropy;

       public:
        using StochasticRescalingManostat::StochasticRescalingManostat;

        [[nodiscard]]
        linalg::tensor3D calculateMu(double volume) override;
    };

    /**
     * @brief AnisotropicStochasticRescalingManostat inherits from
     * StochasticRescalingManostat
     *
     * _(https://doi.org/10.1063/5.0020514)
     *
     */
    class AnisotropicStochasticRescalingManostat
        : public StochasticRescalingManostat
    {
       public:
        using StochasticRescalingManostat::StochasticRescalingManostat;

        [[nodiscard]]
        linalg::tensor3D calculateMu(double volume) override;
    };

    /**
     * @brief FullAnisotropicStochasticRescalingManostat inherits from
     * StochasticRescalingManostat
     *
     * _(https://doi.org/10.1063/5.0020514)
     *
     */
    class FullAnisotropicStochasticRescalingManostat
        : public StochasticRescalingManostat
    {
       public:
        using StochasticRescalingManostat::StochasticRescalingManostat;

        [[nodiscard]]
        linalg::tensor3D calculateMu(double volume) override;
    };

}   // namespace manostat

#endif   // _STOCHASTIC_RESCALING_MANOSTAT_HPP_
