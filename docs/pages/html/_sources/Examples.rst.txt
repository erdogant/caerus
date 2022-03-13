Detect valleys and peaks in stockmarket data
##############################################

Facebook
*********

In the following example we load the 2016 elections data of the USA for various candidates.
We will check whether the votes are fraudulent based on benfords distribution.

.. code:: python

	from caerus import caerus

	# Initialize
	bl = caerus(alpha=0.05)

	# Load elections example
	df = bl.import_example(data='USA')

	# Extract election information.
	X = df['votes'].loc[df['candidate']=='Donald Trump'].values

	# Print
	print(X)
	# array([ 5387, 23618,  1710, ...,    16,    21,     0], dtype=int64)

	# Make fit
	results = bl.fit(X)

	# Plot
	bl.plot(title='Donald Trump')


.. |fig1| image:: ../figs/figure_fb.png

.. table:: Facebook stockmarket data
   :align: center

   +----------+
   | |fig1|   |
   +----------+


Bitcoin
*********


.. |fig2| image:: ../figs/figure_btc.png
.. |fig4| image:: ../figs/figure_btc_last_300days.png

.. table:: Bitcoin trend
   :align: center

   +----------+
   | |fig2|   |
   +----------+
   | |fig4|   |
   +----------+



Gridsearch
##############################################

xxxx


.. |fig3| image:: ../figs/figure_gridsearch_btc.png

.. table:: Bitcoin trend
   :align: center

   +----------+
   | |fig3|   |
   +----------+


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
