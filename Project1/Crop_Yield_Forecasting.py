import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import gaussian_process

def clean_data(filename):
  df    = pd.DataFrame.from_csv("Data_Sets/"+filename,parse_dates=False,infer_datetime_format=False)
  index = df.index
  new_index = []
  for element in index:
    if isinstance(element, basestring):
      temp = float(element.replace(",",""))
    else:
      temp = element
    new_index.append(int(round(temp)))
  df.index      = new_index
  df.index.name = 'Year'
  for col in df.columns:
    if 'Unnamed' in col:
        del df[col]
  return df

def gather_all_data(filenames):
  df = pd.DataFrame()
  for filename in filenames:
    df = pd.concat([df, clean_data(filename)], axis=1)
  return df

def main():
  filenames = ['Soil_Moisture_From_Plot.csv', 'Rain_Fall_From_Plot.csv', 'NDVI_From_Plot.csv', 'Surface_Temp_From_Plot.csv']
  input  = gather_all_data(filenames)
  output = clean_data('Crop_Yield_From_Plot.csv')
  input  = input[input.index != 1994]
  output = output[output.index != 1994]

  Predicted_Output = pd.DataFrame(index=output.index, columns=['Predicted Yield','Percent Error'])
  for Year_In_Question in output.index:

    X = np.array(input[input.index != Year_In_Question])
    x = np.array(input[input.index == Year_In_Question])
    Y = np.ravel(output[output.index != Year_In_Question])

# Theil Sen Regressor regression
    predict_ = x
    gp = linear_model.TheilSenRegressor(tol=0.0001)
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'TheilSen'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'TheilSen Percent Error'] = float(np.abs(output.loc[Year_In_Question, 'Crop Yield'] - y_pred)/output.loc[Year_In_Question, 'Crop Yield']*100)

# Linear regression
    predict_ = x
    gp = linear_model.LinearRegression()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Linear'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Linear Percent Error'] = float(np.abs(output.loc[Year_In_Question, 'Crop Yield'] - y_pred)/output.loc[Year_In_Question, 'Crop Yield']*100)

# Polynomial
    poly     = PolynomialFeatures(degree=2)
    X_       = poly.fit_transform(X)
    predict_ = poly.fit_transform(x)
    gp = linear_model.LinearRegression()
    gp.fit(X_, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Polynomial'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Polynomial Percent Error'] = float(np.abs(output.loc[Year_In_Question, 'Crop Yield'] - y_pred)/output.loc[Year_In_Question, 'Crop Yield']*100)

# Ridge Regression
    predict_ = x
    gp = linear_model.Ridge()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Ridge'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Ridge Percent Error'] = float(np.abs(output.loc[Year_In_Question, 'Crop Yield'] - y_pred)/output.loc[Year_In_Question, 'Crop Yield']*100)

# Lasso Regression
    predict_ = x
    gp = linear_model.Lasso()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Lasso'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Lasso Percent Error'] = float(np.abs(output.loc[Year_In_Question, 'Crop Yield'] - y_pred)/output.loc[Year_In_Question, 'Crop Yield']*100)

# Elastic Net
    predict_ = x
    gp = linear_model.ElasticNet()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Elastic'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Elastic Percent Error'] = float(np.abs(output.loc[Year_In_Question, 'Crop Yield'] - y_pred)/output.loc[Year_In_Question, 'Crop Yield']*100)

# Bayesian Ridge Modeling
    predict_ = x
    gp = linear_model.BayesianRidge()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'BayRidge'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'BayRidge Percent Error'] = float(np.abs(output.loc[Year_In_Question, 'Crop Yield'] - y_pred)/output.loc[Year_In_Question, 'Crop Yield']*100)

# Gaussian Processes for Machine Learning (GPML)
    predict_ = x
    gp = gaussian_process.GaussianProcess()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'GPML'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'GPML Percent Error'] = float(np.abs(output.loc[Year_In_Question, 'Crop Yield'] - y_pred)/output.loc[Year_In_Question, 'Crop Yield']*100)

  output = output.join(Predicted_Output)
# print output

# Plot the results!

  fig = pl.figure()
  pl.plot(input.index, output['Crop Yield'], 'r-.', label=u'$Observations$')

  pl.plot(input.index, output['Linear'], 'k:', label=u'Linear')
# pl.plot(input.index, output['Polynomial'], 'm-', label=u'Polynomial p=2')
  pl.plot(input.index, output['GPML'], 'g-', label=u'GPML')
  pl.plot(input.index, output['TheilSen'], 'b-', label=u'Theil Sen')
  pl.plot(input.index, output['Ridge'], 'k-', label=u'Ridge')
  pl.plot(input.index, output['Lasso'], 'c-', label=u'Lasso')
  pl.plot(input.index, output['Elastic'], 'y-', label=u'Elastic Net')
  pl.plot(input.index, output['BayRidge'], 'b:', label=u'Bayesian Ridge')

  pl.xlabel('$Year$')
  pl.ylabel('$Crop Yield$')
  pl.xlim(1982, 2004)
  pl.legend(loc='best')
  pl.savefig('Crop_Yield_Prediciton.png', bbox_inches='tight')
  pl.show()


  fig = pl.figure()
  pl.plot(input.index, output['Linear Percent Error'], 'k:', label=u'Linear')
# pl.plot(input.index, output['Polynomial Percent Error'], 'm-', label=u'Polynomial p=2')
  pl.plot(input.index, output['GPML Percent Error'], 'g-', label=u'GPML')
  pl.plot(input.index, output['TheilSen Percent Error'], 'b-', label=u'Theil Sen')
  pl.plot(input.index, output['Ridge Percent Error'], 'k-', label=u'Ridge')
  pl.plot(input.index, output['Lasso Percent Error'], 'c-', label=u'Lasso')
  pl.plot(input.index, output['Elastic Percent Error'], 'y-', label=u'Elastic Net')
  pl.plot(input.index, output['BayRidge Percent Error'], 'b:', label=u'Bayesian Ridge')

  pl.xlabel('$Year$')
  pl.ylabel('$\% Error$')
  pl.xlim(1982, 2004)
  pl.legend(loc='best')
  pl.savefig('Crop_Yield_Percent_Error.png', bbox_inches='tight')
  pl.show()



if __name__ == '__main__':
  main()
