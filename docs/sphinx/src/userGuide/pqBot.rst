.. _pqbot:

######
PQ Bot
######

PQ Bot handles small repository tasks and pull request reviews. GitHub may not
offer the App account in the ``@`` autocomplete list; typing ``@pq-bot`` still
works.

Requesters must have repository write access. Coworker changes arrive
as draft pull requests for human review. PQ Bot runs only while the
repository is public. The bot never merges.

Commands
********

    | ``@pq-bot test <area>`` - add or extend unit tests.
    | ``@pq-bot fix #<n>`` - propose a small fix for an open issue.
    | ``@pq-bot repro #<n>`` - reproduce and repair a small open issue.
    | ``/pq-bot review`` - advisory review of the current pull request.

Start a comment line with the command. Commands inside quoted text or
code fences are ignored. Coworker commands run from issue comments; reviews
run from pull request comments. Other commands are ignored.

Pull request reviews
********************

On a pull request, start a comment line with ``/pq-bot review`` or
``@pq-bot review``. Requesters need write access to the repository.
PQ Bot reads a bounded text diff and posts an
advisory ``COMMENT`` review from the existing GitHub App account. It
does not approve, request changes, or alter the pull request branch.
Reviews use ``PQ_BOT_MODEL_REVIEW`` by default.

Test-driven fixes
*****************

``fix`` and ``repro`` use two separate model runs. The first may edit only a
single supported unit-test family. The trusted runner verifies that the
repository passes before the test and fails after it. The second run receives
the failure output, may not alter the frozen test, and writes the smallest
implementation that makes it pass. The trusted runner then requires the same
test suite to pass before publication.

The model never runs tests or Git commands. C++ unit tests under ``tests/`` and
Python unit tests under ``scripts/tests/`` are supported by this flow. Tasks
that cannot express the behavior in one of those suites stop without a pull
request. C++ coverage extends an existing registered test source; Python test
files may be added because the Python suite discovers them automatically.

``test`` uses one model run and accepts only added C++ or Python unit-test
coverage plus one changelog fragment. The trusted runner requires the
corresponding test suite to pass before publication.

Model choice
************

Append ``with <name>`` to pick a model, e.g.
``@pq-bot fix #123 with smart`` or ``/pq-bot review with smart``.
Names map to repository variables. Reviews without a model name use
``PQ_BOT_MODEL_REVIEW``. Coworker tasks use ``PQ_BOT_MODEL_CHEAP``
by default. Unknown model names are rejected.

What to expect
**************

The workflow must exist on the default branch to receive comment events.
It needs ``OPENCODE_API_KEY``, ``PQ_BOT_APP_ID``, and
``PQ_BOT_PRIVATE_KEY`` secrets and the model variables above. OpenCode
can edit only a disposable copy without a GitHub write token. A
separate validator limits the diff to 12 files and 100 changed lines,
runs the applicable trusted tests, and exports only the validated file
contents and task context. A second job starts on a fresh runner, checks
that bundle again against the original comment and unchanged ``dev``
commit, and only then creates the GitHub App token used to open a draft PR
from a ``pq-bot/`` branch. Model-generated code and repository write
credentials therefore never share a runner. Human review and CI decide
whether the change merges.
The performance gate is skipped for bot branches; a human must run that
check on a trusted branch before merging a performance-related change. The
required changelog fragment supplies the plain-language summary in a short,
structured PR description. Larger tasks need a human contributor.
