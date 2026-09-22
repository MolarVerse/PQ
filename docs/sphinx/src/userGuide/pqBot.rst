.. _pqbot:

######
PQ Bot
######

PQ Bot is a coworker bot for repository tasks. Mention ``@pq-bot``
in an issue or pull request comment and it picks up the work.

Only MolarVerse members and owners can trigger it. All results
arrive as pull requests for human review. The bot never merges.

Commands
********

    | ``@pq-bot test <area>`` - add or extend unit tests.
    | ``@pq-bot fix #<n>`` - resolve an issue.
    | ``@pq-bot review`` - advisory line review of the pull request.
    | ``@pq-bot cleanup <path>`` - narrow tidy-ups.
    | ``@pq-bot triage`` - label the thread or ask for repro info.
    | ``@pq-bot repro #<n>`` - reproduce a reported bug.

Anything outside these tasks is refused.

Model choice
************

Append ``with <name>`` to pick a model, e.g.
``@pq-bot review with smart``. Names map to repository variables;
without a name the account default is used.

What to expect
**************

Work lands on a ``pq-bot/`` branch targeting ``dev``, with tests
and a changelog fragment, following the contributor rules. A
requested reviewer still needs to approve before anything merges.
