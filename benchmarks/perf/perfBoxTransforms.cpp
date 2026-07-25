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

// Fixed-work micro-benchmark of the box coordinate transforms: wrapping into
// the box, and triclinic <-> orthogonal space conversions.

#include <cstdio>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "orthorhombicBox.hpp"
#include "triclinicBox.hpp"
#include "vector3d.hpp"

static constexpr long ITERATIONS = 20000;

int main()
{
    using linearAlgebra::Vec3D;

    auto ortho = simulationBox::OrthorhombicBox();
    ortho.setBoxDimensions({20.0, 20.0, 20.0});

    auto triclinic = simulationBox::TriclinicBox();
    triclinic.setBoxDimensions({20.0, 20.0, 20.0});
    triclinic.setBoxAngles({80.0, 90.0, 100.0});

    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (long i = 0; i < ITERATIONS; ++i)
    {
        const double x = (i & 127) * 0.3 - 19.0;
        const Vec3D  v(x, 0.5 * x, -x);

        sink += norm(ortho.wrapPositionIntoBox(v));
        sink += norm(triclinic.wrapPositionIntoBox(v));
        sink += norm(triclinic.toOrthoSpace(v));
        sink += norm(triclinic.toSimSpace(v));
    }

    std::printf("%.6f\n", sink);
    return 0;
}
