import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500],
    'price': [200000, 300000, 400000, 500000]
})

X = data[['area']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[1800]])
print("Predicted price:", prediction[0])