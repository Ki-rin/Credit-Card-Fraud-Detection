import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from jedi.api.refactoring import inline
from sklearn.model_selection import train_test_split
from scipy import stats

warnings.filterwarnings(action='once')
mpl.style.use('ggplot')
sns.set(style='whitegrid')


def classification_eval(estimator, X_test, y_test):
    from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, cohen_kappa_score, \
        accuracy_score, average_precision_score, roc_auc_score
    y_pred = estimator.predict(X_test)
    dec = np.int64(np.ceil(np.log10(len(y_test))))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred), '\n')

    print('Classification report')
    print(classification_report(y_test, y_pred, digits=dec))

    print("Scalar Metrics")
    format_str = '%%13s = %%.%if' % dec
    print(format_str % ('MCC', matthews_corrcoef(y_test, y_pred)))

    if y_test.nunique() <= 2:  # Additional metrics for binary classification
        try:
            y_score = estimator.predict_proba(X_test)[:, 1]
        except:
            y_score = estimator.decision_function(X_test)
        print(format_str % ('AUPRC', average_precision_score(y_test, y_score)))
        print(format_str % ('AUROC', roc_auc_score(y_test, y_score)))
    print(format_str % ("Cohen's kappa", cohen_kappa_score(y_test, y_pred)))
    print(format_str % ('Accuracy', accuracy_score(y_test, y_pred)))

if __name__ == '__main__':

    # Read Data
    transactions = pd.read_csv(r"C:\Users\Miriam\PycharmProjects\pythonProject\creditcard.csv")
    print("shape : ", transactions.shape)
    print("info : ",transactions.info())
    print("is null any : ",transactions.isnull().any().any()) #Check for any missing data in CSV file.
    print("head : \n", transactions.head())

    # Display the frequency of fraudulent transactions. 1 stands for fradulent and 0 for true.
    print("count values for each class : ",transactions['Class'].value_counts())
    print("normalize counts : ",transactions['Class'].value_counts(normalize = True))

    # Train/Test Split

    X = transactions.drop(labels='Class', axis=1)  # Features
    y = transactions.loc[:, 'Class']  # Response
    del transactions #Delete the original data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    print("shapes of test/train", X_train.shape, X_test.shape)

    # to flag off warnings
    X_train.is_copy = False
    X_test.is_copy = False


    # Data Analysis
    print("Describe time : ", X_train['Time'].describe())

    # convert seconds to hours for easy of calculations
    X_train.loc[:, 'Time'] = X_train.Time / 3600
    X_test.loc[:, 'Time'] = X_test.Time / 3600

    # time of last transaction in days
    print("time of last transaction in days: ", X_train['Time'].max() / 24)

    # histogram of transition times.
    plt.figure(figsize=(12, 8))
    sns.distplot(X_train['Time'], bins=50, color='green')
    plt.xlim([0, 50])
    plt.xticks(np.arange(0, 50, 5))
    plt.xlabel('Time after 1st transaction(hr)')
    plt.ylabel('Count')
    plt.title('Transaction times')

    # Summary stats
    print("Sums by amount: ",X_train['Amount'].describe())

    # histogram
    plt.figure(figsize=(12, 8))
    sns.distplot(X_train['Amount'], bins=5000, color='g')
    plt.xlim([0, 1000])
    plt.xticks(np.arange(0, 500, 50))
    plt.ylabel('Count')
    plt.title('Transaction Amounts')

    # box plot as the histogram doesnot shoe the details properly.
    plt.figure(figsize=(12, 8), dpi=80)
    sns.boxplot(X_train['Amount'])
    plt.title('Transaction Amounts')

    print("skewness:",X_train['Amount'].skew())

    #Lets remove the skewness and convert the data into a normal distribution.
    X_train.loc[:, 'Amount'] = X_train['Amount'] + 1e-9     # Shift all amounts by 1e-9

    #Lets remove the skewness and convert the data into a normal distribution.
    X_train.loc[:,'Amount'], maxlog, (min_ci, max_ci) = sp.stats.boxcox(X_train['Amount'], alpha=0.01)

    #The maximum likelihood estimate of  λ  in the Box-Cox transform:

    X_train.dropna()

    # plotting newly transformed accounts
    plt.figure(figsize=(12, 8))
    sns.distplot(X_train['Amount'], color='g')
    plt.xlabel('Transformed Amount')
    plt.ylabel('Count')
    plt.title('Transaction Amounts (Box-Cox Transformed)')

    print("Sums by amount: ",X_train['Amount'].describe())
    print("Skewness: ",X_train['Amount'].skew())

    '''Power transform removed most of the skewness in the Amount variable. Now we need to compute 
    the Box-Cox transform on the test data amounts as well, using the  λ  value estimated on the training data.
    '''
    X_test.loc[:, 'Amount'] = X_test['Amount'] + 1e-9  # Shift all amounts by 1e-9
    X_test.loc[:, 'Amount'] = sp.stats.boxcox(X_test['Amount'], lmbda=maxlog)

    # Time vs Amount
    sns.jointplot(x=X_train['Time'].apply(lambda x: x % 24), y=X_train['Amount'], kind='hex', height=12,
                  xlim=(0, 24), ylim=(-7.5, 14)).set_axis_labels('Time of Day (hr)', 'Transformed Amount')

    #Let's compare the descriptive stats of the PCA variables V1-V28.
    pca_vars = ['V%i' % k for k in range(1, 29)]
    print("Let's compare the descriptive stats of the PCA variables V1-V28 :",X_train[pca_vars].describe())

    plt.figure(figsize=(12, 8))
    sns.barplot(x=pca_vars, y=X_train[pca_vars].mean(), color='green')
    plt.xlabel('Column')
    plt.ylabel('Mean')
    plt.title('V1-V28 Means')

    #All of V1-V28 have approximately zero mean. Now plot the standard deviations:
    plt.figure(figsize=(12, 8))
    sns.barplot(x=pca_vars, y=X_train[pca_vars].std(), color='green')
    plt.xlabel('Column')
    plt.ylabel('Std Dev')
    plt.title('V1-V28 Std Dev')

    #The PCA variables have roughly unit variance, but as low as ~0.3 and as high as ~1.9. Plot the skewnesses next:
    plt.figure(figsize=(12, 8))
    sns.barplot(x=pca_vars, y=X_train[pca_vars].skew(), color='green')
    plt.xlabel('Column')
    plt.ylabel('Skew')
    plt.title('V1-V28 Skewness')

    #Mutual Information between Fraud and the Predictors
    from sklearn.feature_selection import mutual_info_classif

    data = mutual_info_classif(X_train, y_train, discrete_features=False, random_state=1)
    mutual_infos = pd.Series(data, index=X_train.columns)

    #The calculated mutual informations of each variable with Class, in descending order:
    print("mutual info:",mutual_infos.sort_values(ascending=False))

    #Logistic Regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier

    pipeline_sgd = Pipeline([
        ('scaler', StandardScaler(copy=False)),
        ('model', SGDClassifier(max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
    ])
    param_grid_sgd = [{
        'model__loss': ['log'],
        'model__penalty': ['l1', 'l2'],
        'model__alpha': np.logspace(start=-3, stop=3, num=20)
    }, {
        'model__loss': ['hinge'],
        'model__alpha': np.logspace(start=-3, stop=3, num=20),
        'model__class_weight': [None, 'balanced']
    }]
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, matthews_corrcoef

    MCC_scorer = make_scorer(matthews_corrcoef)
    grid_sgd = GridSearchCV(estimator=pipeline_sgd, param_grid=param_grid_sgd, scoring=MCC_scorer, n_jobs=-1,
                            pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)
    import warnings

    with warnings.catch_warnings():  # Suppress warnings from the matthews_corrcoef function
        warnings.simplefilter("ignore")
        grid_sgd.fit(X_train, y_train)

    print("grid_sgd.best_score_",grid_sgd.best_score_)
    print("grid_sgd.best_params_",grid_sgd.best_params_)

    #Random Forest
    from sklearn.ensemble import RandomForestClassifier

    pipeline_rf = Pipeline([
        ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
    ])

    param_grid_rf = {'model__n_estimators': [75]}

    grid_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, scoring=MCC_scorer, n_jobs=-1,
                           pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)

    grid_rf.fit(X_train, y_train)

    print("grid_sgd.best_score_",grid_rf.best_score_)
    print("grid_sgd.best_params_", grid_rf.best_params_)



    classification_eval(grid_rf, X_test, y_test)