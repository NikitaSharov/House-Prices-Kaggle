from pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor


train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror')
p = Pipeline(train, test)
x_train, y_train, x_test = p.prepare_data()
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
result = cross_val_score(xgb, X=x_train, y=y_train, cv=kfolds, scoring='neg_root_mean_squared_error')
# predictions = model.predict(x_test)
# output = pd.DataFrame({'Id': Id, 'SalePrice': predictions})
# output.to_csv('submission.csv', index=False)
# os.system('kaggle competitions submit -f submission.csv -m "Baseline" -q house-prices-advanced-regression-techniques')