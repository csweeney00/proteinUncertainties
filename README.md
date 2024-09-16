# proteinUncertainties
Code implementation for Master's Dissertation

This repository contains the code used to run the simple regression and parametric curve regression (ToyDatasets) and also the code for the adapted ABodyBuilder2 model (training_code)

The data files used to train the model are not included in the repository because there are a very large amount of them. However, they can be generated in the following way:
1. Run gen_loops_dataset.py: This splits the beta_sheet_loops dataset into smaller chunks before downloading the pdb files referenced in the split of the dataset specified by the file_num parameter.
2. Run build_loops_dataset.py: This generates the pickled loops dataset, currently set up to only consider loops of length 11.
3. Run prepare_data.py: This processes the pdb files downloaded by step 1 into a format that is taken by the model. It produces the coordinates files for each loop and the all_data.csv file
4. Run train_test_split.py: This splits the loops in all_data.csv into train, validation, and test files.


