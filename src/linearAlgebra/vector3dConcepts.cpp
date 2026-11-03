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

// Header-only; this translation unit exists only so that
// vector3dConcepts.hpp has a same-named compile command in
// compile_commands.json. Without it, clangd cannot infer which flags to
// use when analyzing the header standalone (e.g. in CI's clangd-tidy lint
// job) and falls back to an unrelated compile command with no project
// include paths.
#include "vector3d/concepts/vector3dConcepts.hpp"
