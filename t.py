import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# load dataset
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
Y = dataset[:, 13]
# define base mode
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.save_weights("abc")
    for layer in model.layers:
        g = layer.get_config()
        h = layer.get_weights()
        print (g)
        print (h)
    return model


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=10, batch_size=50, verbose=0)
# fix random seed for reproducibility
epochs = [10] # add 50, 100, 150 etc
batch_size = [100] # add 5, 10, 20, 40, 60, 80, 100 etc
param_grid = dict(epochs=epochs, batch_size=batch_size)


##############################################################
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1,verbose=2)
grid_result = grid.fit(X, Y)

print estimator.model.layers[0].get_weights()[1]
exit()
print grid_result.best_estimator_.layers[0].get_weights()[1]
print grid_result.best_estimator_.layers[1].get_weights()[1]
print grid_result.get_params()
# print grid.
##############################################################
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

#
# seed = 7
# numpy.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
# # use 10-fold cross validation to evaluate this baseline model
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))