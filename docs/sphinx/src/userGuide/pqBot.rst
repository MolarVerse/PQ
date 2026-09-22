.. _pqbot:

######
PQ Bot
######

PQ Bot is a coworker bot for repository tasks. Write ``@pq-bot``
in an issue or pull request comment to request a task. GitHub may not
offer this App account in the ``@`` autocomplete list; typing the text
still works.

Requesters must have repository write access. Coworker changes arrive
as draft pull requests for human review. The bot never merges.

Commands
********

    | ``@pq-bot test <area>`` - add or extend unit tests.
    | ``@pq-bot fix #<n>`` - propose a small fix for an open issue.
    | ``/pq-bot review`` - advisory review of the pull request.
    | ``@pq-bot cleanup <path>`` - narrow tidy-ups.
    | ``@pq-bot format <path>`` - small formatting changes.
    | ``@pq-bot repro #<n>`` - add a focused reproduction test.
    | ``@pq-bot docs <area>`` - propose documentation edits.
    | ``@pq-bot deps <area>`` - propose a small dependency edit.
    | ``@pq-bot perf <area>`` - propose a small performance edit.

Start a comment line with the command. Commands inside quoted text or
code fences are ignored. ``triage``, ``rerun``, and ``rebase`` are not
active commands; they need separate permissions and operational rules.

Pull request reviews
********************

On a pull request, start a comment line with ``/pq-bot review`` or
``@pq-bot review``. Requesters need write access to the repository.
PQ Bot reads a bounded text diff and posts an
advisory ``COMMENT`` review from the existing GitHub App account. It
does not approve, request changes, or alter the pull request branch.
Reviews use ``PQ_BOT_MODEL_REVIEW`` by default.

A separate machine user is optional. If one is configured as
``PQ_BOT_MACHINE_USER``, requesting that user as a reviewer also starts
the same App-backed review. The machine user needs its own GitHub
account and repository access to become selectable in GitHub's
reviewer picker.

Model choice
************

Append ``with <name>`` to pick a model, e.g.
``@pq-bot fix #123 with smart`` or ``/pq-bot review with smart``.
Names map to repository variables. Reviews without a model name use
``PQ_BOT_MODEL_REVIEW``. Coworker tasks use ``PQ_BOT_MODEL_CHEAP``
by default. Unknown coworker model names are rejected.

What to expect
**************

The workflow must exist on the default branch to receive comment events.
It needs ``OPENCODE_API_KEY``, ``PQ_BOT_APP_ID``, and
``PQ_BOT_PRIVATE_KEY`` secrets and the model variables above. OpenCode
can edit only a disposable copy without a GitHub write token. A
separate validator limits the diff to 12 files and 100 changed lines,
runs the repository script tests, then opens a draft PR from a
``pq-bot/`` branch targeting ``dev``. Human review and CI decide whether
the change merges. Larger tasks need a human contributor.
