.. _releasesAndHotfixes:

#####################
Releases and Hotfixes
#####################

Use a normal release to publish the reviewed work in ``dev``. Use a hotfix for
an urgent correction to the version on ``main`` without including other work
from ``dev``. Both paths publish a stable GitHub release and a tag on the merge
commit. Neither path publishes a draft or prerelease.

Replace ``vX.Y.Z``, branch names, issue numbers and commit placeholders in the
commands below with the intended values. Run them from a PQ checkout with Git
and the GitHub CLI authenticated to ``MolarVerse/PQ``.

**************
Normal release
**************

#. Check the work selected for release and choose an unused ``vX.Y.Z``. Regular
   changes contribute fragments under ``changes/user`` or ``changes/developer``;
   do not edit the rendered changelogs by hand.
#. Open **Actions > Create Release PR > Run workflow**, select ``main`` and
   enter the version. The CLI equivalent is:

   .. code-block:: bash

      gh workflow run create-release-pr.yml --repo MolarVerse/PQ \
        --ref main -f version=vX.Y.Z

   This checks tag/branch availability, creates ``release/vX.Y.Z`` from ``dev``,
   and opens a pull request to ``main`` with the title ``Release-X.Y.Z``.
   A failed Git lookup stops creation; it is not treated as an available name.
#. Link the release issue in the pull request description using ``Related to
   #ISSUE_NUMBER`` or a closing keyword with the actual number. Review the
   selected commits and generated changelogs.
#. Wait for the bot's preparation commit and the checks on that revision.
   Opening, reopening, editing or updating the PR runs ``check-release``. It
   records the version, renders the fragments, tests the scripts and verifies
   preparation before committing the result to the release branch.
#. Add any correction as a normal commit with an appropriate fragment on the
   release branch. Preparation incorporates later fragments into the same
   release entry. Existing entries, history and the original release date are
   preserved; retrying an interrupted preparation does not duplicate notes.
#. Merge the reviewed PR after its checks pass. Publication validates and tests
   the merged revision again, then waits for successful ``BUILD`` and ``LINT``
   push runs on ``main`` for that exact SHA. Only after those checks pass is
   the write-enabled release App token minted to create the tag and release.

The **Create Release PR** workflow is dispatched from the selected ref. Changes
merged only into ``dev`` do not replace the production manual workflow on
``main``. The release candidate must contain the intended lifecycle scripts.

********
Hotfixes
********

#. Create the hotfix from the current ``main`` so it contains the released code
   and the correction, without unrelated changes from ``dev``:

   .. code-block:: bash

      git fetch origin main dev
      git switch -c hotfix/short-description origin/main
      # Apply the correction and add a changelog fragment.
      git add path/to/corrected-file changes/user/bugfix.short-description.md
      git commit -m "fix: describe the correction"
      git push -u origin hotfix/short-description

#. Open a PR from that branch to ``main``, link its issue, and request review:

   .. code-block:: bash

      gh pr create --repo MolarVerse/PQ --base main \
        --head hotfix/short-description --title "fix: describe the correction"

#. The first successful preparation chooses the highest numeric stable tag and
   increments its patch number. For example, ``v1.2.3`` becomes ``v1.2.4``.
   Prerelease tags are excluded from this inference. The bot stores the chosen
   version and branch name in ``.github/release.json`` and commits it with the
   prepared changelogs.
#. Review that preparation commit and wait for the checks on it. Later jobs
   reuse the stored version even if newer tags appear. Do not manually bump the
   version between preparation and publication.
#. Merge after review and successful checks. The same post-merge CI gate,
   publication and verification rules apply as for a normal release.

If another publication takes the selected version, stop and choose a new
candidate. Do not delete or retarget the occupied tag. One recovery path is to
create a new hotfix branch from the latest ``origin/main`` and cherry-pick only
the correction commits, including their fragments. Do not cherry-pick the old
bot preparation commit or its release metadata. Open a replacement PR and
close the superseded PR after the replacement is ready.

After a push to ``main``, the sync workflow brings the change back to ``dev``
and then ``NEXT``. Check that sync separately; a published release does not
prove that the branch sync succeeded.

************************
Publication verification
************************

``publish-release`` accepts only a commit already included in ``main``. It
requires successful main-branch push runs of both ``ci_build.yml`` and
``lint.yml`` for the publication SHA, waiting up to 30 minutes. A successful
run for ``dev``, a PR merge ref or another commit cannot satisfy the gate.
Failed, cancelled, missing, skipped, malformed or timed-out results block
publication. A newer failed or queued run takes precedence over an old success.

The publication helper verifies that the remote tag resolves to the expected
commit and that the GitHub release is published and stable. Existing annotated
tags are compared by their underlying commit and are not replaced. An existing
draft, prerelease, conflicting tag or incomplete record requires review.

After the lifecycle reports success:

.. code-block:: bash

   gh release view vX.Y.Z --repo MolarVerse/PQ \
     --json tagName,isDraft,isPrerelease,publishedAt,url
   git ls-remote --tags origin "refs/tags/vX.Y.Z" "refs/tags/vX.Y.Z^{}"

Confirm that the version is correct, both flags are false, a publication time
is present, and the tag's commit (the peeled ``^{}`` entry for an annotated
tag) equals the intended merge SHA. Also check the ``Release Lifecycle`` run
and the separate branch-sync runs. CI success alone does not prove publication.

Repository protection should require the PR checks before merge. The
publication guard prevents a failed build from being released; it does not
prevent a merge into an unprotected branch.

**********************
Failures and recovery
**********************

* **Creation stopped after pushing a release branch:** inspect the branch and
  any existing PR before retrying. The creation workflow refuses an existing
  release branch. If the branch contains the intended candidate and no PR
  exists, open its PR to ``main`` with the exact ``Release-X.Y.Z`` title; do not
  overwrite the branch to make a retry pass.
* **Wrong regular-release title:** change it to exactly ``Release-X.Y.Z``.
  The edited event reruns preparation and validation.
* **Failed preparation or script test:** inspect that step, fix the cause on
  the release/hotfix branch, and push the correction. Existing rendered entries
  and surviving fragments can be replayed after interrupted writes or cleanup.
* **Failed, cancelled or missing post-merge CI:** inspect the ``BUILD`` and
  ``LINT`` runs for the exact merge SHA. Rerun or repair the failing workflow
  before retrying publication. An unrelated green commit does not qualify.
* **API or connection error:** stop and inspect the remote state. Only an HTTP
  404 release lookup means absence. A failed lookup is not permission to create
  or replace an object. If creation succeeded but the response was lost, a
  retry verifies the existing release rather than posting it again.
* **Tag push succeeded but release creation failed:** keep the tag and inspect
  both the tag and release. A retry accepts the existing tag only when it points
  to the same commit. A tag moved during publication produces an error, even if
  the release request already took effect; inspect the partial state manually.
* **Existing draft, prerelease or conflicting tag:** resolve the discrepancy
  explicitly. The helper does not promote a draft or replace a tag for you.

To retry a prepared publication after merging, use the original version and
the full intended merge SHA:

.. code-block:: bash

   gh workflow run release.yml --repo MolarVerse/PQ --ref main \
     -f version=vX.Y.Z -f commit=FULL_MERGE_SHA

Manual recovery uses the same changelog, script-test and exact-commit CI gates.
It does not bypass them. Publication jobs are serialized; check the run's
actual result rather than assuming that a dispatch completed.

The ``restore`` input of **Create Release PR** is for rebuilding a release from
``dev`` after a rollback on ``main``. It constructs a recovery candidate whose
tree matches ``dev``. Use an unused version and review that recovery tree. If
it contains no unreleased notes, add a recovery fragment on the new release
branch so preparation can run. Restore does not delete an existing release,
rewind branches or authorize reusing its tag.

*************************
Testing before publishing
*************************

Run the script suite in Linux with Bash, Git and Python 3.12 or newer:

.. code-block:: bash

   python3 -m unittest discover -s scripts/tests -p 'test_*.py'

On a workstation with Docker, this command tests committed ``HEAD`` in a
disposable container without copying Git credentials, ignored files or local
build directories:

.. code-block:: bash

   git archive HEAD | docker run --rm -i --cpus=2 --memory=2g \
     --workdir /work python:3.12-slim-bookworm bash -c '
       set -euo pipefail
       tar -x
       apt-get update -qq
       apt-get install -y -qq --no-install-recommends git bash coreutils
       python3 -m unittest discover -s scripts/tests -p "test_*.py"
     '

The tests use temporary Git remotes and an HTTP fixture. They cover conflicting
and annotated tags, concurrent publishers, lost creation responses, failed
pushes, malformed API data, pinned hotfix versions and interrupted preparation.
They do not publish a real PQ release. Hosted event delivery, App installation
permissions and scheduler behavior need a separate GitHub-side rehearsal; a
sandbox repository can exercise those without publishing PQ.
