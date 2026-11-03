.. _releasesAndHotfixes:

#####################
Releases and Hotfixes
#####################

These steps use the GitHub website.

**************
Normal release
**************

#. Open **Actions** and select **Create Release PR**.
#. Click **Run workflow**. Choose ``main``, enter an unused version such as
   ``v1.2.3``, leave **restore** unchecked, then click **Run workflow** again.
#. Open the generated PR to ``main``. Link its issue, review and merge once
   checks pass. The release branch starts from ``dev``.

******
Hotfix
******

#. On **Code**, select ``main``. Use the branch dropdown to create
   ``hotfix/short-description`` from it.
#. Make the correction and add a fragment such as
   ``changes/user/bugfix.short-description.md``.
#. Open **New pull request** with **base** ``main`` and **compare** your hotfix
   branch. Link its issue, review and merge once checks pass.

Opening the PR starts **Release Lifecycle** automatically. It selects and
keeps the patch version. After merge, publication waits for BUILD and LINT.

************
Check a run
************

Open **Actions**, then **Release Lifecycle**. Fixing code on the PR branch
starts a new run. For temporary failures, use **Re-run failed jobs**.
Check the version and commit under **Releases** after success.

The **Run workflow** button requires write access and the workflow on ``main``.
Use :doc:`releaseCli` if the GitHub interface is unavailable.

.. toctree::
   :hidden:

   releaseCli
