.. include:: add_top.add

Mythology
###########################

In Greek mythology, Caerus (same as kairos) was the personification of opportunity, luck and favorable moments. He was shown with only one lock of hair. His Roman equivalent was Occasio or Tempus. Caerus was the youngest child of Zeus.

Method
###########################

caerus is a python package providing that determines the local-minima with the corresponding local-maxima within the given time-frame. The method is build using a forward rolling window to iteratively evaluate thousands of windows. For each window a score of percentages is computed from the start-to-stop position. The resulting matrix is a [window x length dataframe] for which only the high scoring percentages, e.g. those above a certain value (minperc) are used. The best scoring percentages is then aggregated by sum per time-point followed by a cut using the threshold. The resulting regions are subsequently detected, and represent the starting-locations of the trade. The stop-locations are determined based on the distance and percentage of te start-locations. As an example, if you want to have best regions, use threshold=1, minperc=high and nlargest=1 (small).

Here are just a few of the things that caerus does well: 
	
	* Detects start-stop regions of local minima and maxima.
	* Insightful plots. 
	* Parameter gridsearch.
	* Designed for the detection of complex trend movements such as in stockmarket data.



.. include:: add_bottom.add