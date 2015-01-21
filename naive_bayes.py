"""
Module creates a Naive Bayes predictor for gender from
Actual & Ideal Weight
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB


def munge_data(csvfile='ideal_weight.csv'):
    """Parse csv file and create data frame"""
    dframe = pd.read_csv(csvfile)
    columns = [name.replace('\'', '') for name in dframe.columns]
    dframe.columns = columns
    dframe['sex'] = dframe['sex'].map(lambda x: 1 if x == '\'Male\'' else 0)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.subplots_adjust(hspace=0.6)
    ax1.hist(dframe['actual'], alpha=0.5, label='Actual')
    ax1.hist(dframe['ideal'], alpha=0.5, label='Ideal')
    ax1.set_title('Actual & Ideal Weight')
    ax1.set_xlabel('Pounds')
    ax1.set_ylabel('Number of People')
    ax1.legend(loc='upper right')
    ax2.hist(dframe['diff'], color='red')
    ax2.set_title('Difference')
    ax2.set_xlabel('Pounds')
    ax2.set_ylabel('Number of People')
    fig.savefig('weight_dist.png')
    pct_men = dframe['sex'].sum() / float(len(dframe['sex']))
    print 'Percentage of men in dataset: {:.2%}'.format(pct_men)
    dframe.drop('id', axis=1, inplace=True)
    x_data = dframe.drop('sex', axis=1)
    y_data = dframe['sex']
    return x_data, y_data


def fit_bayes(x_data, y_data, test_size=0.1):
    """Fit a Naive Bayes Classifier on x_data & y_data"""
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print 'Classifier Score: {:.2%}'.format(clf.score(x_test, y_test))
    preds = clf.predict([[145, 160, -15], [160, 145, 15]])
    preds = ['Male' if x == 1 else 'Female' for x in preds]
    # map(lambda x: 'Male' if x == 1 else 'Female')
    print ('Prediction for 145 lb. person whose ideal '
           'weight is 160: {}').format(preds[0])
    print ('Prediction for 160 lb. person whose ideal '
           'weight is 145: {}').format(preds[1])

if __name__ == '__main__':
    x_vals, y_vals = munge_data()
    fit_bayes(x_vals, y_vals)
