import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline


# Load the competition data
comp_data = pd.read_csv("./competition_data.csv")

# Split into training and evaluation samples
# dividimos el DataFrame original comp_data en dos conjuntos: uno (train_data) que contiene filas con 
# valores nulos en la columna "ROW_ID" y otro (eval_data) que contiene filas con valores no nulos 
# en la misma columna.
train_data = comp_data[comp_data["ROW_ID"].isna()]
eval_data = comp_data[comp_data["ROW_ID"].notna()]
tree = DecisionTreeClassifier(random_state = 22)
scores = cross_val_score(tree, train_data, eval_data, cv = KFold(4))
del comp_data
gc.collect()


###
parameters = {'criterion':('gini', 'entropy', 'log_loss'),
              'splitter': ('best', 'random'),
              'max_depth': list(range(5, 41)),
              'min_samples_split': list(range(2, 21)),
              'min_samples_leaf': list(range(1, 16)),
              'min_impurity_decrease': uniform(loc = 0, scale = 0.1) 
             }

rs = RandomizedSearchCV(estimator = DecisionTreeClassifier(random_state = 22),
                        param_distributions = parameters,
                        n_iter = 1800, # cantidad de iteraciones, sino haria infinitas
                        cv = KFold(4),
                        random_state = 22)

rs.fit(train_data, eval_data)
###


# Train a random forest model on the train data
train_data = train_data.sample(frac=1/3) # toma un tercio de los datos 
y_train = train_data["conversion"]
X_train = train_data.drop(columns=["conversion", "ROW_ID"])
X_train = X_train.select_dtypes(include='number') # solo columnas numericas
del train_data
gc.collect()

cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))
cls.fit(X_train, y_train)
# modelo del árbol de decisión se entrenará en las características de X_train y aprenderá a 
# predecir las etiquetas de y_train utilizando los valores faltantes imputados por el SimpleImputer.

# Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"]) # elimina la columna conversion
eval_data = eval_data.select_dtypes(include='number') # filtra y deja solo los valores numericos
y_preds = cls.predict_proba(eval_data.drop(columns=["ROW_ID"]))[:, cls.classes_ == 1].squeeze() # utiliza el modelo entrenado para hacer las predicciones

# Make the submission file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)
