# names-app

This project is an application of a names classifier, that is, classify a name of either a girl's name of boy's name. It is built as an ilustration of building machine learning applications, but the performance of the classifier itself is of no interest.

Up to now, this only works with spanish names.

I expect that the particular choice for problem and solution approach is of no real value (from the machine learning perspective), but let me know if you can think of some ;)

## How to use it

The application runs by issuing the following command in the terminal:
` python local_app.py --log <log-level> <command-name> <command-options>
The application has 4 relevant commands:
* `train`: to train the model on a given dataset.
* `test`: to test the model on a given dataset.
* `infer`: to infer whether a given name is that of a girl or a boy. Use a quoted name in case of names with spaces on it.
* `download-data`: to download the dataset of names, parse, split the data into train and test sets and store it as .hdf5 file.

Please use the `--help` option for more details.
