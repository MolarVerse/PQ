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

*****************
How to Contribute
*****************

For any contributor willing to contribute to the project, it is important to understand the branching model used by the project. The project uses the `Gitflow <http://nvie.com/posts/a-successful-git-branching-model/>`_ branching model. In order to contribute to the project please follow the following steps:


    #. Fork the project on Github. (not necessary if you are a member of the project)

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
    
    #. Create a pull request on Github.
