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

 a. Create a new environment from the YAML specification

    .. code-block:: console

        conda env create -f docs/env/conda_env_py36_[osx64|i686].yml

*or*

 b. Create a new environment from scratch and let ``pip`` figure out dependencies and you sort out potential issues

    .. code-block:: console

        conda create -n plasticc
        source activate plasticc
        [pip install six] # you might need this if you get errors about six not being available
        pip install -r requirements.txt


3. Run the notebooks:
~~~~~~~~~~~~~~~~~~~~~

Start the notebook

.. code-block:: console

    jupyter notebook plasticc_astro_starter_kit.ipynb


4. Troubleshooting:
~~~~~~~~~~~~~~~~~~~

If you still get errors about missing packages when you run the notebook
 - make sure you executed the ``source activate plasticc`` command in step 2.
 - check that ``jupyter`` is using the right kernel with the ``Kernel > Change Kernel`` menu item and selecting the one with ``plasticc`` in the name

If for some reason you can't find a kernel with ``plasticc`` in the name, then inside the ``plasticc`` environment:

 a. install ``ipykernel`` and register the ``plasticc`` environment:

    .. code-block:: console

        conda install ipykernel
        python -m ipykernel install --user --name plasticc --display-name "Python (plasticc)"

*or*

 b. install the ``nb_conda`` package and let ``conda`` manage integration with Jupyter

    .. code-block:: console

        source deactivate
        conda install nb_conda
        source activate plasticc

