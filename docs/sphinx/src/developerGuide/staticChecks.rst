.. _staticChecks:

#############################
Static Checks and Conventions
#############################

A pull request has to pass several checks that look at the source code without running it: the formatting, the license
header, the compiler warnings, ``clangd-tidy`` and the *DevOps C++ checks*. This page lists what each of them wants, how to
run it on your machine and the mistakes that cost the most round trips. The workflows that run them are described on
the :doc:`continuous integration <continuousIntegration>` page.

.. list-table::
    :header-rows: 1
    :widths: 24 30 46

    * - Check
      - Configured in
      - Applies to
    * - clang-format 20.1.3
      - ``.clang-format``
      - ``apps``, ``benchmarks``, ``include``, ``src``, ``tests``
    * - License header
      - ``config/licenseHeader.txt``
      - every ``.cpp``, ``.hpp``, ``.c``, ``.h`` file in those directories
    * - Compiler warnings
      - ``CMakeLists.txt``
      - all targets of the project (strict set in Debug builds)
    * - clangd-tidy
      - ``.clang-tidy``, ``tests/.clang-tidy``, ``benchmarks/.clang-tidy``
      - the changed C++ files, tests and benchmarks included
    * - DevOps C++ checks
      - ``devops.toml``
      - ``src``, ``include``, ``tests``, ``benchmarks``, ``apps``
    * - Commit message prefix
      - ``.githooks/commit-msg``
      - every commit, locally and in the review

**********
Formatting
**********

The code is formatted with **clang-format 20.1.3** (Google style as a base: 4 spaces, 80 columns, Allman braces; see
``.clang-format``). CI checks every C++ file that the pull request touches **as a whole**, so a file with a formatting
difference in a line you did not change still fails.

The version matters. A newer clang-format formats differently: version 21, which many Linux distributions already ship,
writes ``const T& x`` where version 20.1.3 keeps the repository's ``const T &x``. Install the pinned version next to
your system one:

.. code:: bash

    $ python3 -m venv ~/.venvs/clang-format
    $ ~/.venvs/clang-format/bin/pip install clang-format==20.1.3
    $ ~/.venvs/clang-format/bin/clang-format --style=file -i <file>

``clang-format --style=file <file> | diff <file> -`` shows what CI would complain about without changing the file. Do
not switch formatting off with ``// clang-format off``: the DevOps checks reject it.

**************
License header
**************

Every C++ file starts with the exact header of ``config/licenseHeader.txt`` (a byte-for-byte comparison). ``bash
scripts/addLicense.sh`` adds it to new files and repairs existing ones.

*****************
Compiler warnings
*****************

Every build uses ``-Wall -Wextra -Wswitch-enum`` and turns a ``switch`` over an enum that misses a value into an
error. **Debug builds** (also the Debug jobs of CI) additionally use a strict set with ``-Werror``:

.. code:: text

    -Wpedantic -Werror -Wundef -Wshadow -Wconversion -Wsign-conversion -Wcast-align -Wformat=2
    -Wnull-dereference -Wdouble-promotion -Wimplicit-fallthrough -Woverloaded-virtual -Wnon-virtual-dtor
    -Wold-style-cast -Wmisleading-indentation -Wuninitialized -Winit-self
    GCC only: -Wuseless-cast -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wmaybe-uninitialized

A Release build does not use this set, so a warning can appear first in CI. Build Debug once before you push a change
that touches numerics or conversions. The tests and the ``benchmarks`` are compiled with the same flags as the library.

***********
clangd-tidy
***********

The LINT workflow runs ``scripts/clang_tidy.sh`` on the changed C++ files. The checks are configured in
``.clang-tidy`` (almost everything enabled, with a list of disabled checks and the reason for each); the files
``tests/.clang-tidy`` and ``benchmarks/.clang-tidy`` inherit it and additionally disable the magic number checks.

.. code:: bash

    $ python3 -m pip install clangd-tidy          # and clangd-20
    $ scripts/clang_tidy.sh -p build              # changed files since origin/dev; build/ holds compile_commands.json
    $ scripts/clang_tidy.sh -p build --all        # every tracked C++ file
    $ scripts/clang_tidy.sh -p build --recheck    # only the files that had findings in the last run
    $ scripts/clang_tidy.sh -p build --base <sha> --head <sha>

The report of the last run is ``clangd-tidy-report.log``. Configure with ``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON``. The
finding you will meet most often is *Statement should be inside braces*
(``google-readability-braces-around-statements``): every ``if``, ``for`` and ``while`` body gets braces, also a single
``throw``.

.. _runningCppChecks:

*****************
DevOps C++ checks
*****************

``external/devops`` (a submodule) provides the ``cpp_checks`` command that CI runs. It first applies text rules to the
files and then parses them with libclang to apply the *AST checks* below. The configuration is ``devops.toml`` in the
repository root.

.. code:: bash

    $ git submodule update --init external/devops
    $ pip install -e "external/devops[ast]"
    $ cmake -S . -B .build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_NATIVE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DBUILD_WITH_TESTS=ON -DBUILD_WITH_DOCS=OFF -DBUILD_WITH_PERF_BENCH=ON -DBUILD_WITH_BENCHMARKING=ON
    $ cpp_checks --base-ref origin/dev       # only the files changed since origin/dev, as CI does
    $ cpp_checks --no-fail-fast              # all files, every finding instead of the first failing file

``.build`` is where ``devops.toml`` expects the compile database (``ast_check_compile_commands_db``). A finding looks
like this:

.. code:: text

    include/thermostat/noseHooverThermostat.hpp:47:14: style: private member function 'requireCoupling' should
        start with a leading underscore ('_requireCoupling') [memberFunctionLeadingUnderscore]

On some machines libclang reports ``astParse`` for a header it cannot parse because it does not find the compiler's own
headers; CI is the reference in that case. VS Code can run ``cpp_checks`` for you, see :ref:`the editor tasks
<editorTasks>`.

The enabled AST checks (``ast_check_enabled_ids`` in ``devops.toml``):

.. list-table::
    :header-rows: 1
    :widths: 28 72

    * - Id
      - What it wants
    * - ``memberLeadingUnderscore``, ``memberFunctionLeadingUnderscore``
      - Private and protected member variables and member functions start with ``_``. Constructors, destructors,
        operators and functions that override a virtual function are exempt.
    * - ``noPublicLeadingUnderscore``
      - Public members do *not* start with ``_``.
    * - ``classMemberOrder``
      - In a class the member variables come first (public, protected, private), then the member functions (public,
        protected, private). A private helper therefore goes to the end of the class, after the public setters and
        getters. The test macros and ``MSTD_ENUM`` are ignored.
    * - ``paramNameForType``
      - A parameter of type ``molsys::SimulationBox`` is named ``simulationBox`` or ``other``; one of type
        ``physicalData::PhysicalData`` is named ``physicalData`` or ``other``.
    * - ``macroReplacement``
      - Banned macros with their replacement. In the tests ``EXPECT_THROW`` is not allowed, use
        ``EXPECT_THROW_MSG`` (from ``throwWithMessage.hpp``): it also checks the message.
    * - ``noGlobalUsing``, ``noGlobalUsingEnum``
      - No ``using namespace`` and no ``using enum`` at namespace or global scope (inside a function body it is fine).
    * - ``noThrowParen``
      - ``throw expression;`` and not ``throw (expression);``.
    * - ``noNewInGtestSetup``, ``noDeleteInGtestTeardown``
      - No ``new`` in a fixture's ``SetUp`` and no ``delete`` in its ``TearDown``; use smart pointers or members.
    * - ``noFinalKeyword``
      - No ``final`` on classes or methods, so that tests can derive from them (issue #802).
    * - ``enforceTypenameOverClass``
      - ``template <typename T>`` and not ``template <class T>``.
    * - ``enforceSystemHeaderLibs``
      - ``mstd`` is included with angle brackets (``#include <mstd/...>``), like a system library (issue #682).
    * - ``includeIwyuPragmaOnly``
      - A comment behind an ``#include`` may only be an IWYU pragma (``export``), no free text (issue #623).
    * - ``noClangFormatToggle``
      - No ``// clang-format off`` and ``// clang-format on``.

**************************
Commit messages and merges
**************************

``.githooks/commit-msg`` rejects a commit whose subject does not start with one of the conventional prefixes
(``feat:``, ``fix:``, ``docs:``, ``test:``, ``ci:``, ``refactor:``, ``merge:``, ... the full table is in ``AGENTS.md``).
Install it once per clone with ``cp .githooks/commit-msg .git/hooks/``. The hook also checks the message Git proposes
for a merge, which starts with ``Merge branch ...`` and is rejected. Give the merge a message of its own:

.. code:: bash

    $ git merge origin/dev -m "merge: dev into feature/my-feature"

.. _editorTasks:

**********************
Editor tasks (VS Code)
**********************

``.vscode/tasks.json`` defines tasks that run the checks and show the findings in the Problems panel:
``clangd-tidy (changed)``, ``clangd-tidy (all)``, ``clangd-tidy (recheck)``, ``devops cpp_checks (changed)`` and
``devops cpp_checks (all)``. They need the setup of the sections above (``clangd-tidy``, ``cpp_checks`` and a compile
database).

*********************
Conventions for tests
*********************

    - ``tests/src`` mirrors the layout of ``src`` and ``include``; a new class gets its tests in the matching module.
    - Check exceptions with ``EXPECT_THROW_MSG(statement, ExceptionType, "message")``.
    - Helpers of a test fixture follow the same naming rule as production code: a protected helper starts with ``_``.
    - The settings classes hold global state (``settings::TimingsSettings::setTimeStep`` and similar). A test that
      changes a setting must restore it, otherwise it changes the result of a test that runs later in the same
      executable. Do it in ``TearDown`` or at the end of the test.
    - Write the test that fails first when you fix a bug, and check that it fails without the fix. Weigh a new guard
      against what the input validation already ensures: many "division by zero" candidates are impossible because the
      input parser rejects the value.
