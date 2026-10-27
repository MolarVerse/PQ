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
The only exception is the CI metrics collector, which commits the data below ``.github/ci-metrics/data/`` to ``dev``,
see :doc:`ciMetrics`.

A typical change:

.. code:: bash

    $ git checkout dev && git pull
    $ git checkout -b feature/my-feature
    # ... commit with a conventional prefix ("feat: ...", "fix: ...", "docs: ...") ...
    $ git push -u origin feature/my-feature

Open the pull request against ``dev``. It needs ``Closes #N``, ``Fixes #N``, ``Resolves #N`` or ``Related to #N`` in the
description, and a changelog fragment (see :ref:`changelogFragments`). If ``dev`` moves while the pull request is open,
update the branch: the ``pr_request`` check fails otherwise.

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

A release is a pull request from ``release/vX.Y.Z`` to ``main`` with ``Release-X.Y.Z`` in the title or body, created
with the **Create Release PR** workflow (``create-release-pr.yml``, started by hand with the version as ``vX.Y.Z``). It
creates the branch from ``dev``, opens the pull request and requests the reviews.

The **Release Lifecycle** workflow (``release.yml``) then takes over:

    #. ``prepare-changelogs``: runs ``scripts/update_changelog.py`` on the release branch. It renders every changelog
       fragment of ``changes/user`` into ``CHANGELOG.md`` and of ``changes/developer`` into ``DEV-CHANGELOG.md``,
       stamps the release header, deletes the fragments and commits the result. Nobody edits the changelogs by hand.
    #. ``check-release``: runs the tests of the scripts and verifies that the changelogs are prepared and that no
       fragment is left.
    #. ``publish-release``: when the pull request is merged, creates the tag ``vX.Y.Z`` on the merge commit and the
       GitHub release with generated notes.

A **hotfix** uses a branch ``hotfix/<name>`` instead; its version is the latest tag with the patch number incremented.
After every push to ``main`` the sync workflow brings the change back to ``dev``.

If a release has to be undone, the ``restore`` input of **Create Release PR** rebuilds the release from ``dev`` after a
rollback.

.. note::

    A regular pull request never edits ``CHANGELOG.md`` or ``DEV-CHANGELOG.md``. It only adds a fragment.
