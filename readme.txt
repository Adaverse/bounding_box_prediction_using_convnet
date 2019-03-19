We are using Tensorflow Estimator API with cusom model and loss function.

Our source code in .ipynb file(recommmended) and also in .py file.

There are certain extra files like hfliphorizontal.csv and vflipvertical.csv files. These contians the coordinates
of the flipped images in both the directions.

path - you need to change this to the folder where all the images are present. It is in 
the first training cell in the first line of code

We are reading .csv files that are in the .zip file the last 4 cells so the path must be
changed to the upadated path. It is present within pd.read_csv(...)

In the last cell last line we save the predicted file in a path that needs to changed too.

In the training cells numbered as 1,2,3 have the tf.estimator.Estimator funcion call where
the model_dir is needed to be set. That is to be changed too in the all the 3 training cells.

Thank You for your patience. The code structure could have been much better.