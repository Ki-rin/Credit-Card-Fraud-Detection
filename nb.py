import pandas as pd
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



warnings.filterwarnings(action='once')



if __name__ == '__main__':

    # Read Data
    transactions_raw = pd.read_csv(r"C:\Users\Miriam\PycharmProjects\pythonProject\creditcard.csv",header=0)
    print(transactions_raw.head())

    #Normalise
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(transactions_raw)
    transactions = pd.DataFrame(scaled)
    transactions.dropna()
    print(transactions.head())

    # Train/Test Split
    X = transactions.drop(labels=30, axis=1)  # Features
    y = transactions.loc[:, 30]  # Response
    print(sum(y))
    #del transactions #Delete the original data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)


    # to flag off warnings
    X_train.is_copy = False
    X_test.is_copy = False

    #Naive Bayes
    from sklearn.naive_bayes import ComplementNB

    classifier = ComplementNB()
    classifier.fit(X_train, y_train)

    y_hat = classifier.predict(X_test)

    wrong = [(p,e) for (p,e) in zip(y_hat, y_test) if p != e]
    print("wrong: ", wrong)
    print(f'{classifier.score(X_test,y_test):.2%}')

    from sklearn.metrics import accuracy_score
    print("Accuracy Score")
    print(accuracy_score(y_test, y_hat))


    from sklearn.metrics import confusion_matrix
    print("Confusion Matrix")
    print(confusion_matrix(y_test,y_hat))


    from sklearn.metrics import classification_report
    print("Classification Report")
    print(classification_report(y_test, y_hat))


