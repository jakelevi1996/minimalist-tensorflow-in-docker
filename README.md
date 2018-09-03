# tensorflow-docker-example

Minimalist example of running a simple Tensorflow program in a docker container, with CPU and GPU options

## TODO

- Make a good README
- Add docstrings to all function
- Write Dockerfiles and scripts to automate building and running them
- Optionally visualise predictions through course of training
- Rename repo as `minimalist-tensorflow-in-docker`
- Tidy up filename manipulation (including *optional* timestamps in filenames)
- Add usage description to README.md
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
- Update `plot_results` in `inference.py` to not be sensitive to the size of the test set