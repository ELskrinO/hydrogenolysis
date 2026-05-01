import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score

from mpl_toolkits.mplot3d import Axes3D
import pickle

# Считываем файл
data = pd.read_excel('data_new.xlsx')

# Выделяем X и Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Выводим матрицу корреляций
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('correlation')
plt.savefig('correlation_matrix.jpg')
plt.show()
#Создаем гистрограммы распределения 
data.hist()
plt.show()

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Модель МНК
model_MNK = LinearRegression()
model_MNK.fit(X_train, Y_train)

# Предсказываем значения на обучающей выборке
Y_pred_MNK = model_MNK.predict(X_train)

#предказываем значене на тестовыой выборке
Y_pred_test = model_MNK.predict(X_test)

# Считаем MAE МНК
MAE_MNK = mean_absolute_error(Y_test, Y_pred_test)
print("MAE МНК_test:", MAE_MNK)

# Проводим кросс-валидацию МНК
cvtest= RepeatedKFold( n_splits=5, n_repeats=10)  
cross_val_scores_MNK = cross_val_score(model_MNK, X, Y, scoring='neg_mean_absolute_error', cv=cvtest, n_jobs=1, error_score='raise')

# Выводим MAE кросс-валидации МНК на экран 
print("MAE кросс-валидации MNK:", cross_val_scores_MNK)

# Переводим значения MAE в положительные
cross_val_scores_MNK = -cross_val_scores_MNK


# Выводим среднее значение кросс-валидации на экран
print("Среднее значение кросс-валидации_MNK:", np.mean(cross_val_scores_MNK))

print ("Среднеквадратичное отклонение_cross_MNK:", cross_val_scores_MNK.std())

# Модель дерева решений
model_DT = DecisionTreeRegressor(max_depth=3)
model_DT.fit(X, Y)

# Предсказываем значения на обучающей выборке
Y_pred_DT = model_DT.predict(X)

# Считаем MAE дерева решений
MAE_DT = mean_absolute_error(Y, Y_pred_DT)
print("MAE дерева решений:", MAE_DT)

# Сохраняем дерево решений в формате Graphviz
export_graphviz(model_DT, out_file='decision_tree.txt',filled=True, rounded=True, feature_names=X.columns)

# Модель случайного леса
model_RF = RandomForestRegressor()
model_RF.fit(X_train, Y_train)

# Предсказываем значения на тестовой выборке
Y_pred_RF = model_RF.predict(X_test)

# Предсказываем значение на обучающей выборке
Y_pred_RF_train = model_RF.predict(X_train)

# Считаем MAE случайного леса на тестовых данных
MAE_RF = mean_absolute_error(Y_test, Y_pred_RF)
MAE_RF = round(MAE_RF, 1)
print("MAE случайного леса на тестовых данных:", MAE_RF)

# Считаем MAE случайного леса на обучающих данных
MAE_RF_train = mean_absolute_error(Y_train, Y_pred_RF_train)
MAE_RF_train = round(MAE_RF_train, 1)
print("MAE случайного леса на обучающих данных:", MAE_RF_train)

# Получаем важности параметров из модели случайного леса
importances = model_RF.feature_importances_

# Создаем список пар (название параметра, важность)
feature_importances = list(zip(X.columns, importances))

# Сортируем список по убыванию важности
feature_importances.sort(key=lambda x: x[1], reverse=True)

# Выводим наиболее важные параметры на экран
indices = np.argsort(importances)[::-1]
for feature, importance in feature_importances:
    print(feature, ":", importance)

#выделяем 3 наиболее важных параметра
top_3_indices = indices[:3]
labels = X.columns[top_3_indices]


#Сохраняем наиболее важные параметры в файл формата txt
with open('important_parameters.txt', 'w') as file:
    for feature, importance in feature_importances:
        file.write(f"{feature}: {importance}") 

# Проводим кросс-валидацию
cvtest= RepeatedKFold( n_splits=5, n_repeats=10, random_state=42)  
cross_val_scores = cross_val_score(model_RF, X, Y, scoring='neg_mean_absolute_error', cv=cvtest, n_jobs=1, error_score='raise')

# Выводим MAE кросс-валидации на экран
print("MAE кросс-валидации:", cross_val_scores)

# Переводим значения MAE в положительные
cross_val_scores = -cross_val_scores

# Переобозначаем кросс валидацию с считаем его среднее значения округляя его до десятых
CVS = np.mean(cross_val_scores)
CVS = round(CVS, 1)


# Выводим среднее значение кросс-валидации на экран
print("Среднее значение кросс-валидации:", np.mean(cross_val_scores))

print ("Среднеквадратичное отклонение:", cross_val_scores.std())

# Используем cross_val_score для кросс-валидации
scores = cross_val_score(model_RF, X, Y, scoring='r2', cv=cvtest, n_jobs=1, error_score='raise')

# Вычисляем среднее значение R^2
mean_r2 = np.mean(scores)
print(f'Среднее значение R^2: {mean_r2:.4f}')


dis=Y.std()

print("AAD:", dis)

P = dis/cross_val_scores.mean()

print ("AAD/MAE:",P)

# Вычисляем R2 у тренеровочной
r2_train = r2_score(Y_train, model_RF.predict(X_train))
r2_train = round(r2_train, 2)
print("R2_train: ", r2_train)

# Вычисляем R2 у тестовой 
r2_test = r2_score(Y_test, model_RF.predict(X_test))
r2_test = round(r2_test, 2)
print("R2_test: ", r2_test)

"""
# Строим график предсказанных значений от истинных
plt.figure()
plt.scatter(Y_test, model_RF.predict(X_test), color='blue', label='Тестовые данные')
plt.scatter(Y_train, model_RF.predict(X_train), color='red', label='Обучающие данные')
plt.plot(Y, Y, color='black')
plt.xlabel('CH4_observed')
plt.ylabel('CH4_predicted')
plt.text(25,270 , f'MAE(training dataset) = {MAE_RF_train}', color='red' )
plt.text(25,290, f'R²(training dataset) = {r2_train}', color='red' )
plt.text(25,310, f'MAE(test dataset) = {MAE_RF}', color='blue' )
plt.text(25,330, f'R²(test dataset) = {r2_test}', color='blue' )
plt.text(25,350, f'MAE(Cross-valodation) = {CVS}', color='black' )
plt.tight_layout()
plt.savefig('predictions.jpeg')
"""
# Строим гистограмму важности параметров
plt.figure()
plt.bar(range(len(X.columns)), importances[indices], color = 'blue' )
plt.xticks(range(len(X.columns)), X.columns[indices], rotation='vertical')
plt.xlabel('Feature paremeters')
plt.ylabel('importans of Feature paremeters')
plt.tight_layout()
plt.savefig('importances.jpeg')

#3-д график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sizes = [y for y in Y]
ax.scatter(X.iloc[:, top_3_indices[0]], X.iloc[:, top_3_indices[1]], X.iloc[:, top_3_indices[2]],c=Y , cmap='rainbow')
ax.set_xlabel(labels[0])
ax.set_ylabel(labels[1])
ax.set_zlabel(labels[2])
plt.show()

#эксперимент

# Считываем файл
data = pd.read_excel('EXP.xlsx')

# Выделяем X и Y
X1 = data.iloc[:, :-1]
Y1 = data.iloc[:, -1]

Y23 = model_RF.predict(X1)

#print(Y23)

# Строим график предсказанных значений от истинных
plt.figure()
plt.scatter(Y_train, model_RF.predict(X_train), color='red', label='Обучающие данные')
plt.scatter(Y_test, model_RF.predict(X_test), color='blue', label='Тестовые данные')
plt.scatter(Y1, Y23, color='Green', label='Эксперименты')
plt.plot(Y_test, Y_test, color='black')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.text(-12,0, f'MAE(training dataset) = {MAE_RF_train}', color='red' )
plt.text(-12,-0.5, f'R²(training dataset) = {r2_train}', color='red' )
plt.text(-12,-1, f'MAE(test dataset) = {MAE_RF}', color='blue' )
plt.text(-12,-1.5, f'R²(test dataset) = {r2_test}', color='blue' )
plt.text(-12,-2, f'MAE(Cross-valodation) = {CVS}', color='black' )
plt.text(-12,-3, f'(Experiment)', color='green' )
plt.text(-12,-2.5, f'MAE(Cross-valodation) = {CVS}', color='black' )
#plt.tight_layout()
plt.show()
plt.savefig('EXP.jpeg')



# Сохранение данных 2д графика в формате xlsx
graph_data1 = pd.DataFrame({'Истинные значения тест': Y_test, 'Предсказанные значения тест': model_RF.predict(X_test)})
graph_data2 = pd.DataFrame({'Истинные значения трейн': Y_train, 'Предсказанные значения трейн': model_RF.predict(X_train)})
graph_data3 = pd.DataFrame({'Истинные значения эксперимент': Y1, 'Предсказанные значения эксперимент': Y23})
graph_data1.to_excel('graph1.xlsx', index=False)
graph_data2.to_excel('graph2.xlsx', index=False)
graph_data3.to_excel('graph3.xlsx', index=False)

#сохранение модели
with open('model.pkl', 'wb') as f:
    pickle.dump(model_RF, f)




print("End")
