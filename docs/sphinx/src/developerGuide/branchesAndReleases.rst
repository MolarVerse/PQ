.. _branchesAndReleases:

#####################
Branches and Releases
#####################

********
Branches
********

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Branch
      - Purpose
    * - ``main``
      - The stable branch. It only changes through a release or a hotfix pull request.
    * - ``dev``
      - The integration branch. Every feature and bugfix pull request targets it.
    * - ``NEXT``
      - Kept in sync with ``dev`` automatically (``dev`` is merged into it) by the sync workflow below.
    * - ``feature/<name>``, ``bugfix/<name>``
      - Your work, cut from an up to date ``dev``.

Nobody pushes to ``main`` or ``dev`` directly, and a branch that other people use is never amended or force-pushed.

A typical change:

.. code:: bash

    $ git checkout dev && git pull
    $ git checkout -b feature/my-feature
    # ... commit with a conventional prefix ("feat: ...", "fix: ...", "docs: ...") ...
    $ git push -u origin feature/my-feature

Open the pull request against ``dev``. It needs ``Closes #N``, ``Fixes #N``, ``Resolves #N`` or ``Related to #N`` in the
description, and a changelog fragment (see :ref:`changelogFragments`). If ``dev`` moves while the pull request is open,
update the branch before relying on checks for an older revision.

.. code:: bash

    $ git fetch origin
    $ git merge origin/dev -m "merge: dev into feature/my-feature"

****************
Automatic merges
****************

    - ``sync-dev-into-next.yml`` merges ``dev`` into ``NEXT`` after every push to ``dev`` and once a day. On a conflict
      it opens a pull request from a ``merge/dev-into-NEXT-<time>`` branch for a person to resolve.
    - ``sync-main-into-dev.yml`` merges ``main`` back into ``dev`` after every push to ``main`` (a release or a hotfix),
      so that the version commits and changelogs are on ``dev`` too. On a conflict it opens a pull request.

Both push with a token of a GitHub App, because a push made with the default ``GITHUB_TOKEN`` would not start the CI of
the branch.

********
Releases
********

Normal releases use ``release/vX.Y.Z`` branches prepared from ``dev``. Hotfixes
use ``hotfix/<name>`` branches prepared from ``main``. Both target ``main`` and
require successful checks on the exact publication commit before publishing.

See :doc:`releasesAndHotfixes` for the creation commands, version selection,
changelog preparation, publication verification and recovery procedures.
