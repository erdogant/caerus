.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import caerus

    # Retrieve URLs of malicous and normal urls:
    X, y = caerus.load_example()

    # Learn model on the data
    model = caerus.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = caerus.plot(model)


Installation
''''''''''''

Create environment
------------------


If desired, install ``caerus`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_caerus python=3.6
    conda activate env_caerus


Install via ``pip``:

.. code-block:: console

    # The installation from pypi is disabled:
    pip install caerus

    # Install directly from github
    pip install git+https://github.com/erdogant/caerus


Uninstalling
''''''''''''

If you want to remove your ``caerus`` installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. caerus should be listed.
   conda env list

   # Remove the caerus environment
   conda env remove --name caerus

   # List all the active environments. caerus should be absent.
   conda env list
