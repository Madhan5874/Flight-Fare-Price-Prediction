import pandas as pd

data=pd.read_csv('flight_data.csv')

data.head()

data.keys()

data.dtypes

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

data['encode_airline']=encoder.fit_transform(data['Airline'])

data.drop_duplicates('Airline')

data['encode_source']=encoder.fit_transform(data['Source'])

data.drop_duplicates('Source')

data['encode_Destination']=encoder.fit_transform(data['Destination'])

data.drop_duplicates('Destination')

data['encode_Total_Stops']=encoder.fit_transform(data['Total_Stops'])

data.drop_duplicates('Total_Stops')

x=data[['encode_airline','encode_source','encode_Destination','encode_Total_Stops']]

y=data['Price']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x_train,y_train)

model.predict(x_test)

prediction=model.predict(x_test)

table=pd.DataFrame({'actual':y_test,'predicted':prediction})

table

from sklearn.metrics import r2_score

r2_score(y_test,prediction)

from sklearn.tree import DecisionTreeRegressor

model_1=DecisionTreeRegressor()

model_1.fit(x_train,y_train)

predict=model_1.predict(x_test)

r2_score(y_test,predict)

import pickle
pickle.dump(model_1,open('flight_fare_prediction.pkl','wb'))
x_train.keys()