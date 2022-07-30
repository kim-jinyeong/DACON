import pandas as pd
from icecream import ic
from sklearn.linear_model import LinearRegression

class Baseline1():

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.test = None

    def hook(self):
        self.preprocessing()
        self.model()
        self.submission()

    def preprocessing(self):
        train = pd.read_csv('./dataset/train.csv')
        test = pd.read_csv('./dataset/test.csv')


        train = train.fillna(0)
        train['Date'] = train['Date'].apply(self.get_date)
        train['IsHoliday'] = train['IsHoliday'].apply(self.holiday_to_number)

        test = test.fillna(0)
        test['Date'] = test['Date'].apply(self.get_date)
        test['IsHoliday'] = test['IsHoliday'].apply(self.holiday_to_number)

        train = train.drop(columns=['id'])
        self.test = test.drop(columns=['id'])

        self.x_train = train.drop(columns=['Weekly_Sales'])
        self.y_train = train['Weekly_Sales']

    def get_date(self, date):
        newDate = date[6:10]
        newDate += date[3:5]
        newDate += date[0:2]
        return int(newDate)

    def holiday_to_number(self, isholiday):
        number = 1 if isholiday == True else 0
        return number

    def model(self):
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)
        prediction = model.predict(self.test)
        return prediction

    def submission(self):
        sample_submission = pd.read_csv('./dataset/sample_submission.csv')
        sample_submission['Weekly_Sales'] = self.model()
        sample_submission.to_csv('./content/submission.csv', index=False)
        ic(sample_submission)






