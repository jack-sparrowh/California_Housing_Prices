# California_Housing_Prices

## Work still in progress
* All new updates are made in the second branch "classes_for_pipeline".
* There are still a lot of things missing in this project. Almost all the necessary classes for preprocessing of the data are done, so the final preproc. pipeline can be constructed. Than the modeling part will roll in. All the things are tested and written in my jupyter notebook and after i am ~100% sure that they will work and catch errors i'am adding them here. 
* The main notebook with EDA and the walkthrough of my ideas is not finished, since everything is added and tested in my spare time.
* Final Preprocessing Pipeline was added, it is missing descriptions, but it is able to take any subset of the data and preprocess it according to the figure below (which is yet to be created).

* To this point classes_for_pipeline contains:
  1. CappedTargetDropper
  2. FeaturesAdder
  3. DataDropper
  4. small_PCA
  5. MultiCollinearityHandler
  6. small_Pipeline
  7. PreprocessingPipeline

# Description
Data used  in this repository comes from the StatLib repository. This dataset is based on data from the 1990 California census. The original dataset appeared in Kelley Pace, R., &; Barry, R. (1997). "Sparse spatial autoregressions". Statistics &amp; Probability Letters, 33(3), 291–297. 
