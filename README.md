# tensorflow-docker-example

Minimalist example of running a simple Tensorflow program in a docker container, with CPU and GPU options. *This is a work-in-progress (that recently hasn't seen much progress as I've been working on other projects); this read-me will be updated with more detail in the future, including sections on Installation, Usage, Test, Useful Links.*

## File structure description

The file-structure of this repository is described below, and was inspired by [this blog post by Morgan](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3) on [MetaFlow](https://blog.metaflow.fr/):

- **Top level directory:**
    - Read-me
    - General `git` and IDE-related files
    - `dockerfile`s and `shell`-scripts to automate their building and running
    - `requirements.txt`
    - **Code folder:**
        - `training.py`: where the magic happens
        - `inference.py`: for performing inference using pre-trained models
        - `main.py`: main script for specifying, model-descriptions, training them, saving them, and then using them for inference
        - **data folder:**
            - `data_util.py`: module with functions for hard coded classification-rules for synthetic data, generating synthetic data, and loading synthetic data
            - Data files for training and evaluation, in the `.npz` file format for storing multiple named `Numpy` arrays
        - **models folder:**
            - `classifier.py`: module containing the `class` describing the class of classifier-models used in this repository
            - Folders containing saved models, saved using [`tf.train.Saver.save()`](https://www.tensorflow.org/api_docs/python/tf/train/Saver#save)
        - **results folder:**
            - [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)-files saved using [`tf.summary.FileWriter.add_summary()`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter#add_summary)

## TODO

- Make a good README
- Add usage description to README.md
- Add output images to README
- Add docstrings to all functions
- Write Dockerfiles and scripts to automate building and running them
- Tidy up filename manipulation (including *optional* timestamps in filenames)
- Add accuracy to classifier and training modules
- Add regularisation to classifer module
- Add option to load module for continued training in training module
- Update package to use `tf.data` API
- Fix bug for repeated training in `for` loop
- Add option to train for number of seconds (instead of number of epochs)
- Test saving and restoring of model for different/unknown model params
- Add batching to inputs
- Add dropout
- `load_data`: Add option to generate data if it doesn't exist
- Add option to train on MNIST
- Reuse gradients for tensorboard summaries, instead of computing new ones
- Fix training loop behaviour if `print_every` is not a multiple of `log_every`