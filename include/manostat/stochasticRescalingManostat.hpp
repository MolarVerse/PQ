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

#include "manostat.hpp"                // for Manostat
#include "randomNumberGenerator.hpp"   // for RandomNumberGenerator

namespace manostat
{
    /**
     * @class StochasticRescalingManostat inherits from Manostat
     *
     * @link https://doi.org/10.1063/5.0020514
     *
     */
    class StochasticRescalingManostat : public Manostat
    {
       protected:
        randomNumberGenerator::RandomNumberGenerator _randomNumberGenerator{};

        double              _tau;
        double              _compressibility;
        double              _dt;
        settings::FixedAxis _fixedAxis;

       public:
        StochasticRescalingManostat() = default;
        explicit StochasticRescalingManostat(
            const double              targetPressure,
            const double              tau,
            const double              compressibility,
            const settings::FixedAxis fixedAxis
        );
        ~StochasticRescalingManostat() override = default;

        // copy constructor and copy assignment needed for random number
        // generator
        StochasticRescalingManostat(const StochasticRescalingManostat &other);
        StochasticRescalingManostat &operator=(
            const StochasticRescalingManostat &other
        );
        StochasticRescalingManostat(StochasticRescalingManostat &&) noexcept =
            delete;
        StochasticRescalingManostat &operator=(StochasticRescalingManostat &&
        ) noexcept = delete;

        void applyManostat(
            simulationBox::SimulationBox &simBox,
            physicalData::PhysicalData   &physData
        ) override;

        [[nodiscard]] virtual linearAlgebra::tensor3D calculateMu(const double);

        [[nodiscard]] settings::ManostatType getManostatType() const override;
        [[nodiscard]] settings::Isotropy     getIsotropy() const override;

        [[nodiscard]] double getTau() const;
        [[nodiscard]] double getCompressibility() const;
    };

    /**
     * @class SemiIsotropicStochasticRescalingManostat inherits from Manostat
     *
     * @link https://doi.org/10.1063/5.0020514
     *
     */
    class SemiIsotropicStochasticRescalingManostat
        : public StochasticRescalingManostat
    {
       private:
        size_t              _2DAnisotropicAxis;
        std::vector<size_t> _2DIsotropicAxes;

       public:
        explicit SemiIsotropicStochasticRescalingManostat(
            const double               targetPressure,
            const double               tau,
            const double               compressibility,
            const size_t               anisotropicAxis,
            const std::vector<size_t> &isotropicAxes,
            const settings::FixedAxis  fixedAxis
        );

        [[nodiscard]]
        linearAlgebra::tensor3D calculateMu(const double volume) override;

        [[nodiscard]] settings::Isotropy getIsotropy() const final;
    };

    /**
     * @class AnisotropicStochasticRescalingManostat inherits from Manostat
     *
     * @link https://doi.org/10.1063/5.0020514
     *
     */
    class AnisotropicStochasticRescalingManostat
        : public StochasticRescalingManostat
    {
       public:
        using StochasticRescalingManostat::StochasticRescalingManostat;

        [[nodiscard]]
        linearAlgebra::tensor3D calculateMu(const double volume) override;

        [[nodiscard]] settings::Isotropy getIsotropy() const final;
    };

    /**
     * @class FullAnisotropicStochasticRescalingManostat inherits from Manostat
     *
     * @link https://doi.org/10.1063/5.0020514
     *
     */
    class FullAnisotropicStochasticRescalingManostat
        : public StochasticRescalingManostat
    {
       public:
        using StochasticRescalingManostat::StochasticRescalingManostat;

        [[nodiscard]]
        linearAlgebra::tensor3D calculateMu(const double volume) override;

        [[nodiscard]] settings::Isotropy getIsotropy() const final;
    };

}   // namespace manostat

#endif   // _STOCHASTIC_RESCALING_MANOSTAT_HPP_
