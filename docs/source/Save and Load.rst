.. _code_directive:

-------------------------------------

Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``caerus`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`caerus.save` and function :func:`caerus.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`caerus.save`:

.. code:: python

    import caerus

    # Load example data
    X,y_true = caerus.load_example()

    # Learn model
    model = caerus.fit_transform(X, y_true, pos_label='bad')

    Save model
    status = caerus.save(model, 'learned_model_v1')



Loading
----------------------

Loading a learned model can be done using the function :func:`caerus.load`:

.. code:: python

    import caerus

    # Load model
    model = caerus.load(model, 'learned_model_v1')
