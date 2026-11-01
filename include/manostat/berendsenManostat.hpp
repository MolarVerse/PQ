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

#ifndef MANOSTAT__BERENDSEN_MANOSTAT_HPP
#define MANOSTAT__BERENDSEN_MANOSTAT_HPP

#include "manostat.hpp"

namespace manostat
{
    /**
     * @brief BerendsenManostat inherits from Manostat
     *
     * _(https://doi.org/10.1063/1.448118)
     *
     */
    class BerendsenManostat : public Manostat
    {
       public:
        using Manostat::Manostat;

        void applyManostat(
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData
        ) override;

        [[nodiscard]] virtual linalg::tensor3D calculateMu() const;
    };

    /**
     * @brief SemiIsotropicBerendsenManostat inherits from BerendsenManostat
     *
     * _(https://doi.org/10.1063/1.448118)
     *
     */
    class SemiIsotropicBerendsenManostat : public BerendsenManostat
    {
       public:
        using BerendsenManostat::BerendsenManostat;

        [[nodiscard]] linalg::tensor3D calculateMu() const override;
    };

    /**
     * @brief AnisotropicBerendsenManostat inherits from BerendsenManostat
     *
     * _(https://doi.org/10.1063/1.448118)
     *
     */
    class AnisotropicBerendsenManostat : public BerendsenManostat
    {
       public:
        using BerendsenManostat::BerendsenManostat;

        [[nodiscard]] linalg::tensor3D calculateMu() const override;
    };

    /**
     * @brief FullAnisotropicBerendsenManostat inherits from BerendsenManostat
     *
     * _(https://doi.org/10.1063/1.448118)
     *
     * @details Full anisotropic means that also the angles between the axes are
     * scaled not only the lengths
     *
     */
    class FullAnisotropicBerendsenManostat : public BerendsenManostat
    {
       public:
        using BerendsenManostat::BerendsenManostat;

        [[nodiscard]] linalg::tensor3D calculateMu() const override;
    };

}   // namespace manostat

#endif   // MANOSTAT__BERENDSEN_MANOSTAT_HPP
