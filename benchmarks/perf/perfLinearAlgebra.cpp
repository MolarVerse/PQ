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

// Fixed-work micro-benchmark of the linear-algebra primitives (Vec3D and the
// 3x3 tensor) that underlie every force/energy kernel.

#include <cstdio>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "staticMatrix.hpp"
#include "vector3d.hpp"

static constexpr long ITERATIONS = 20000;

int main()
{
    using namespace linearAlgebra;

    const Vec3D v1{1.1, 2.2, 3.3};
    const Vec3D v2{0.7, -1.3, 2.1};

    // non-singular so inverse() is well defined
    const StaticMatrix3x3<double> matrix{
        Vec3D{2.0, 0.1, 0.2},
        Vec3D{0.3, 3.0, 0.1},
        Vec3D{0.2, 0.1, 4.0}
    };

    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (long i = 0; i < ITERATIONS; ++i)
    {
        const double scale = 1.0 + static_cast<double>(i & 255) * 0.01;
        const Vec3D  a     = v1 * scale;
        const Vec3D  b     = v2 - a;

        sink += norm(a + b) + normSquared(a) + dot(a, b) + norm(cross(a, b));

        const Vec3D matrixVec = matrix * b;
        const auto  matrixSq  = matrix * transpose(matrix);

        sink += norm(matrixVec) + det(matrixSq) + det(inverse(matrix));
    }

    std::printf("%.6f\n", sink);
    return 0;
}
