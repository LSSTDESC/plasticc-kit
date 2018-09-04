=========================
Installation Instructions
=========================

Here's a minimal set of instructions to get up and running with the PLAsTiCC starter kit.

0. Install python:
~~~~~~~~~~~~~~~~~~
We recommend using the anaconda python distribution to run this package. If you
don't have it already, follow the instructions `here
<https://conda.io/docs/install/quick.html#linux-miniconda-install>`__

**Make sure you added the conda/bin dir to your path!**

1. Get the code:
~~~~~~~~~~~~~~~~

Clone this repository

.. code-block:: console

   git clone https://github.com/LSSTDESC/plasticc-kit.git
   cd plasticc-kit/

.. _package:

2. Install everything:
~~~~~~~~~~~~~~~~~~~~~~

 a. Create a new environment from the YAML specification (Preferred! All dependencies resolved!)

    .. code-block:: console

      conda env create -f docs/env/conda_env_py36_[osx64|i686].yml

*or*

 b. Create a new environment from scratch (Let pip figure out dependencies and you sort out potential issues)

  .. code-block:: console

    conda create -n plasticc
    source activate plasticc
    [pip install six] # you might need this if you get errors about six not being available
    pip install -r requirements.txt
