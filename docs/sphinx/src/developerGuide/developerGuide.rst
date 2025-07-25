.. _developerGuide:

###############
Developer Guide
###############

This section includes information for developers who want to contribute to the project. It includes information about the project structure, how to run the tests, and how to build the documentation. It also includes information about the project's coding style and how to contribute to the project.

*****************
Project Structure
*****************

*****
Tests
*****

*************
Documentation
*************

This documentation is written as `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ files ``.rst`` and converted to HTML website files by `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ .
The respective source files are located in ``PQ/docs/sphinx/src/``.
In order to compile the files locally you need to install the following Python packages:

    - `Sphinx            <https://pypi.org/project/Sphinx/>`_
    - `sphinx_sitemap    <https://pypi.org/project/sphinx-sitemap/>`_
    - `breathe           <https://pypi.org/project/breathe/>`_
    - `sphinx_rtd_theme  <https://pypi.org/project/sphinx-rtd-theme/>`_

The project is built by running ``make html`` in the folder ``PQ/docs/sphinx/``.
The resulting website can be viewed by opening the file ``PQ/docs/sphinx/_build/html/index.html`` *via* your favorite browser.

*****************
How to Contribute
*****************

For anyone willing to contribute to the project, it is important to understand the branching model used by the project. The project uses the `Gitflow <http://nvie.com/posts/a-successful-git-branching-model/>`_ branching model. In order to contribute to the project, please follow these steps:


    #. Fork the project on GitHub. (not necessary if you are a member of the project)

    #. Clone your fork locally:
    
        .. code:: bash

            $ git clone https://github.com/MolarVerse/PQ.git

    #. Initialize git flow with the following settings (if not specified default settings are used)

        .. code:: bash

            [master] main
            [develop] dev
            [version tag prefix] v

    #. Create a feature branch for your contribution:
    
        .. code:: bash

            $ git flow feature start <feature_branch_name>


    #. Commit your changes to your feature branch and publish your feature branch:
    
        .. code:: bash

            $ git add <files>
            $ git commit -m "commit message"
            $ git flow feature publish <feature_branch_name>
    
    #. Create a pull request on GitHub.
