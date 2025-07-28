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

#ifndef _BERENDSEN_MANOSTAT_HPP_

#define _BERENDSEN_MANOSTAT_HPP_

#include <cstddef>   // for size_t
#include <vector>    // for vector

#include "manostat.hpp"   // for Manostat
#include "typeAliases.hpp"

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
        explicit BerendsenManostat(const double, const double, const double);

        void applyManostat(pq::SimBox &, pq::PhysicalData &) override;

        [[nodiscard]] virtual pq::tensor3D calculateMu() const;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getTau() const;
        [[nodiscard]] double getCompressibility() const;

        [[nodiscard]] pq::ManostatType getManostatType() const final;
        [[nodiscard]] pq::Isotropy     getIsotropy() const override;
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
        SemiIsotropicBerendsenManostat(
            const double,
            const double,
            const double,
            const size_t,
            const std::vector<size_t> &
        );

        [[nodiscard]] pq::tensor3D calculateMu() const override;

        [[nodiscard]] pq::Isotropy getIsotropy() const final;
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

        [[nodiscard]] pq::tensor3D calculateMu() const override;

        [[nodiscard]] pq::Isotropy getIsotropy() const final;
    };

    /**
     * @class FullAnisotropicBerendsenManostat inherits from BerendsenManostat
     *
     * @link https://doi.org/10.1063/1.448118
     *
     * @details Full anisotropic means that also the angles between the axes are
     * scaled not only the lengths
     *
     */
    class FullAnisotropicBerendsenManostat : public BerendsenManostat
    {
       public:
        using BerendsenManostat::BerendsenManostat;

        [[nodiscard]] pq::tensor3D calculateMu() const override;

        [[nodiscard]] pq::Isotropy getIsotropy() const final;
    };

}   // namespace manostat

#endif   // _BERENDSEN_MANOSTAT_HPP_