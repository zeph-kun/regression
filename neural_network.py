from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors

data = datasets.load_digits()
dir(data)

print(f"Il y a {len(data.target)} classes dans data.target = {data.target}")

def plot_digits(start):
    fig = plt.figure(figsize=(10,10))
    cmap = matplotlib.colors.ListedColormap(['red', 'black'])
    for im in range(12):
        plt.subplot(3,4,im+1)
        title = str(start+im) + ":" + str(data.target[start+im])
        plt.title(title)
        plt.imshow(data.images[start+im], cmap=cmap)
        plt.axis('off')
    plt.show()
plot_digits(1000)


y = data.target
x = data.images.reshape((len(data.images), -1))
x.shape

x_train, y_train, x_test, y_test = x[:1347], y[:1347], x[1347:], y[1347:]
mlp_classifier = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='sgd', tol=0.01,n_iter_no_change=30,
random_state=1, alpha=0.0001, learning_rate_init=.1, verbose=True)

mlp_classifier.fit(x_train,y_train)

predictions = mlp_classifier.predict(x_test)
accuracy_score(y_test, predictions)

print(f" y_test[10:20] = {y_test[10:20]}")
print(f"predictions[10:20] = {predictions[10:20]}")