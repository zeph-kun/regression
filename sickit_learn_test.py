import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.model_selection import KFold


data = pd.read_csv('Data/ozone.csv', sep=',')

y = data["MaxO3"]

data = data[
        ['T9', 'T12', 'T15', 'Ne9', 'Ne12', 'Ne15', 'Vx9',
        'Vx12', 'Vx15', 'MaxO3v']
]

def normalization(dataToNormalize):
    columns = dataToNormalize.columns
    for col in columns:
        x = dataToNormalize[[col]].values.astype(float)
        standard_normalization = preprocessing.StandardScaler()
        res = standard_normalization.fit_transform(x)
        dataToNormalize[col]=res


normalization(data)

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
regression_alg = LinearRegression()
regression_alg.fit(x_train, y_train)

train_predictions = regression_alg.predict(x_train)

print(f"Training RMSE = {round(sqrt(mean_squared_error(y_train, train_predictions)),2)}")
print(f"Training R2_score = {round(r2_score(y_train, train_predictions),2)}")

test_predictions = regression_alg.predict(x_test)

print(f"Test RMSE = {round(sqrt(mean_squared_error(y_test, test_predictions)),2)}")
print(f"Test R2_score = {round(r2_score(y_test, test_predictions),2)}")

regression_alg.coef_
regression_alg.intercept_

plt.scatter(y_test, test_predictions, color='black')
plt.title("les prédictions du modèle vs la réalité")
plt.xlabel("les valeurs observées")
plt.ylabel("Les prédictions")
plt.plot([40.0, 160.0], [40.0, 160.0], 'red', lw=1)
plt.show()


def average_result(nb_run):
    average_rmse = 0
    average_r2 = 0
    for i_run in range(nb_run):
        x_train, x_test, y_train, y_test = train_test_split(data,y, test_size=0.2)

        regression_alg = LinearRegression()
        regression_alg.fit(x_train, y_train)

        test_predictions = regression_alg.predict(x_test)

        i_run_rmse = sqrt(mean_squared_error(y_test, test_predictions))
        i_run_r2 = r2_score(y_test, test_predictions)

        print(f"Run {i_run} : RMSE = {round(i_run_rmse,2)} - R2_score = {round(i_run_r2,2)}")

        average_rmse = average_rmse + i_run_rmse
        average_r2 = average_r2 + i_run_r2
    average_rmse = average_rmse / nb_run
    average_r2 = average_r2 / nb_run

    print(f"Moyenne : RMSE = {round(average_rmse,2)} - R2_score = {round(average_r2,2)}")


average_result(10)


kf = KFold(n_splits=2, shuffle=False)

for train_index, test_index in kf.split(data):
    print("Les indices de train_index = ", train_index)
    print("Les indices de test_index = ", train_index)
    print("\n\n")


kf = KFold(n_splits=2, shuffle=True)

for train_index, test_index in kf.split(data):
    print("Les indices de train_index = ", train_index)
    print("Les indices de test_index = ", train_index)
    print("\n\n")


kf = KFold(n_splits=3, shuffle=False)
for train_index, test_index in kf.split(data):
    print("Les indices de train_index = ", train_index.shape[0])
    print("Les indices de test_index = ", test_index.shape[0])
    print("\n\n")


def create_evaluate_model(index_fold, x_train, x_test, y_train, y_test):
    regression_alg = LinearRegression()
    regression_alg.fit(x_train, y_train)
    test_predictions = regression_alg.predict(x_test)
    rmse = sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)
    print(f"Run {index_fold} : RMSE = {round(rmse,2)} - R2_score = {round(r2,2)}")
    return (rmse, r2)


nb_model = 5
kf = KFold(n_splits=nb_model, shuffle=False)

index_fold = 0
average_rmse = 0
average_r2 = 0

for train_index, test_index in kf.split(data):
    x_train, x_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    current_rmse, current_r2 = create_evaluate_model(index_fold, x_train, x_test, y_train, y_test)

    average_rmse = average_rmse + current_rmse
    average_r2 = average_r2 + current_r2

    index_fold = index_fold + 1

average_rmse = average_rmse / nb_model
average_r2 = average_r2 / nb_model

print(f"Moyenne : RMSE = {round(average_rmse,2)} - R2_score = {round(average_r2,2)}")