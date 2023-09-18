# Bayesian Neural Network

The implementation of the BNN is based on the description from [torchbnn](https://github.com/Harry24k/bayesian-neural-network-pytorch) and from [Tutorial 1: Bayesian Neural Networks with Pyro](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html).

## Ho wot use it?

 * The current implementation required a custom `accuracy` function as well as a _training_ dataset to calculate the statistics.
    * The accuracy function, as well as the training dataset, should be provided using the `data` argument in `get_client`.
    * `data` should be a dictionary with the keys: `train_ds`, `batch_size`, and `accuracy_fun`
    * If no accuracy function is provided a custom one, developed for the _iris_ dataset, will be used (and a key error will be raised when used with custom data)
 * The implementation also lacks on calculating the loss of the training-iteration.

## What needs to be improved?

 * To calculate the loss of each epoch.
 * De-couple the torch's `DataLoader` from the model.