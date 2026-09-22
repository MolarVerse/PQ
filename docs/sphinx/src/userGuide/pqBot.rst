.. _pqbot:

######
PQ Bot
######

PQ Bot is a coworker bot for repository tasks. Write ``@pq-bot``
in an issue or pull request comment to request a task. GitHub may not
offer this App account in the ``@`` autocomplete list; typing the text
still works.

Only MolarVerse members and owners can trigger coworker tasks. Those
results arrive as pull requests for human review. The bot never merges.

Commands
********

    | ``@pq-bot test <area>`` - add or extend unit tests.
    | ``@pq-bot fix #<n>`` - resolve an issue.
    | ``/pq-bot review`` - advisory review of the pull request.
    | ``@pq-bot cleanup <path>`` - narrow tidy-ups.
    | ``@pq-bot triage`` - label the thread or ask for repro info.
    | ``@pq-bot repro #<n>`` - reproduce a reported bug.

Anything outside these tasks is refused.

Pull request reviews
********************

On a pull request, start a comment line with ``/pq-bot review`` or
``@pq-bot review``. Requesters need write access to the repository.
PQ Bot reads a bounded text diff and posts an
advisory ``COMMENT`` review from the existing GitHub App account. It
does not approve, request changes, or alter the pull request branch.
The review uses the repository's ``PQ_BOT_MODEL_REVIEW`` model.

A separate machine user is optional. If one is configured as
``PQ_BOT_MACHINE_USER``, requesting that user as a reviewer also starts
the same App-backed review. The machine user needs its own GitHub
account and repository access to become selectable in GitHub's
reviewer picker.

Model choice
************

Append ``with <name>`` to pick a model, e.g.
``@pq-bot fix #123 with smart``. Names map to repository variables;
without a name the account default is used. Reviews always use
``PQ_BOT_MODEL_REVIEW``.

What to expect
**************

Coworker changes land on a ``pq-bot/`` branch targeting ``dev``,
following the contributor rules. A human reviewer still needs to
approve before anything merges.
