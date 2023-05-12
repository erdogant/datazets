
Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``datazets`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`datazets.save` and function :func:`datazets.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`datazets.save`:

.. code:: python

    import datazets

    # Load example data
    X,y_true = datazets.load_example()

    # Learn model
    model = datazets.fit_transform(X, y_true, pos_label='bad')

    Save model
    status = datazets.save(model, 'learned_model_v1')



Loading
----------------------

Loading a learned model can be done using the function :func:`datazets.load`:

.. code:: python

    import datazets

    # Load model
    model = datazets.load(model, 'learned_model_v1')

