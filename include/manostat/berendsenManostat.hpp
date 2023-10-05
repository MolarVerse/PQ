/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _BERENDSEN_MANOSTAT_HPP_

#define _BERENDSEN_MANOSTAT_HPP_

#include "manostat.hpp"   // for Manostat

#include <cstddef>   // for size_t
#include <vector>    // for vector

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Box;             // forward declaration

}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace manostat
{
    /**
     * @class BerendsenManostat inherits from Manostat
     *
     * @link https://doi.org/10.1063/1.448118
     *
     */
    class BerendsenManostat : public Manostat
    {
      protected:
        double _tau;
        double _compressibility;
        double _dt;

      public:
        explicit BerendsenManostat(const double targetPressure, const double tau, const double compressibility);

        void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        [[nodiscard]] virtual linearAlgebra::Vec3D calculateMu() const;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getTau() const { return _tau; }
    };

    /**
     * @class SemiIsotropicBerendsenManostat inherits from BerendsenManostat
     *
     * @link https://doi.org/10.1063/1.448118
     *
     */
    class SemiIsotropicBerendsenManostat : public BerendsenManostat
    {
      private:
        size_t              _2DAnisotropicAxis;
        std::vector<size_t> _2DIsotropicAxes;

      public:
        SemiIsotropicBerendsenManostat(const double               targetPressure,
                                       const double               tau,
                                       const double               compressibility,
                                       const size_t               anisotropicAxis,
                                       const std::vector<size_t> &isotropicAxes)
            : BerendsenManostat(targetPressure, tau, compressibility), _2DAnisotropicAxis(anisotropicAxis),
              _2DIsotropicAxes(isotropicAxes){};

        [[nodiscard]] linearAlgebra::Vec3D calculateMu() const override;
    };

    /**
     * @class AnisotropicBerendsenManostat inherits from BerendsenManostat
     *
     * @link https://doi.org/10.1063/1.448118
     *
     */
    class AnisotropicBerendsenManostat : public BerendsenManostat
    {
      public:
        using BerendsenManostat::BerendsenManostat;

        [[nodiscard]] linearAlgebra::Vec3D calculateMu() const override;
    };

}   // namespace manostat

#endif   // _BERENDSEN_MANOSTAT_HPP_