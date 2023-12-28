


from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

dataset=load_iris()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=10)
knn_classifier.fit(x_train, y_train)
y_pred=knn_classifier.predict(x_test)
correct=0
incorrect=0
for i in range(len(y_test)):
    if y_pred[i]==y_test[i]:
        correct+=1
    else:
        incorrect+=1
print("Correct number of predictions: ", correct)
print("Incorrect number of predictions: ", incorrect)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)

print("Confusion Matrix: \n", confusion_matrix(y_pred, y_test))

