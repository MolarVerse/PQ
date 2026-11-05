############
CLI fallback
############

Use these commands if the GitHub interface is unavailable. Replace the version,
branch and commit placeholders with the intended values.

Create a normal release PR:

.. code-block:: bash

   gh workflow run create-release-pr.yml --repo MolarVerse/PQ \
     --ref main -f version=vX.Y.Z

Create a hotfix branch, make the correction and open its PR:

.. code-block:: bash

   git fetch origin main
   git switch -c hotfix/short-description origin/main
   # Make the correction and add its changelog fragment.
   git add path/to/corrected-file changes/user/bugfix.short-description.md
   git commit -m "fix: describe the correction"
   git push -u origin hotfix/short-description
   gh pr create --repo MolarVerse/PQ --base main \
     --head hotfix/short-description --title "Hotfix-X.Y.Z: describe the correction"

Retry publication of a prepared, merged release:

.. code-block:: bash

   gh workflow run release.yml --repo MolarVerse/PQ --ref main \
     -f version=vX.Y.Z -f commit=FULL_MERGE_SHA

The same CI checks apply. Existing drafts or conflicting tags require review.
