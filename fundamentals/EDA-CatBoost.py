import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# train_df, test_df = catboost.datasets.amazon() # nice datasets with categorical features only :D
train_df = pd.read_csv('/media/lvision/Sabrent/kaggle/2013/amazon-employee-access-challenge/train.csv')
test_df = pd.read_csv('/media/lvision/Sabrent/kaggle/2013/amazon-employee-access-challenge/test.csv')
print(train_df.shape)
print(type(train_df))
print(test_df.shape)
print(type(test_df))

print(train_df.head())
print(test_df.head())


y = train_df['ACTION']
X = train_df.drop(columns='ACTION') # or X = train_df.drop('ACTION', axis=1)

X_test = test_df.drop(columns='id')



SEED = 1

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=SEED)


# from catboost import CatBoostClassifier
# from catboost import CatBoostRegressor
import catboost as ctb
print(ctb.__version__)


params = {'loss_function':'Logloss', # objective function
          'eval_metric':'AUC', # metric
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': SEED
         }
cbc_1 = ctb.CatBoostClassifier(**params)
cbc_1.fit(X_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
          eval_set=(X_valid, y_valid), # data to validate on
          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
          plot=True # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
         );

cat_features = list(range(X.shape[1]))
print(cat_features)


condition = True # here we specify what condition should be satisfied only by the names of categorical features
cat_features_names = [col for col in X.columns if condition]
cat_features = [X.columns.get_loc(col) for col in cat_features_names]
print(cat_features)


params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'cat_features': cat_features,
          'verbose': 200,
          'random_seed': SEED
         }
cbc_2 = ctb.CatBoostClassifier(**params)
cbc_2.fit(X_train, y_train,
          eval_set=(X_valid, y_valid),
          use_best_model=True,
          plot=True
         );


params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'cat_features': cat_features,
          'early_stopping_rounds': 200,
          'verbose': 200,
          'random_seed': SEED
         }
cbc_2 = ctb.CatBoostClassifier(**params)
cbc_2.fit(X_train, y_train, 
          eval_set=(X_valid, y_valid), 
          use_best_model=True, 
          plot=True
         );


params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'cat_features': cat_features,
          'task_type': 'GPU',
          'verbose': 200,
          'random_seed': SEED
         }
cbc_3 = ctb.CatBoostClassifier(**params)
cbc_3.fit(X_train, y_train,
          eval_set=(X_valid, y_valid), 
          use_best_model=True,
          plot=True
         );


params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'cat_features': cat_features,
          'task_type': 'GPU',
          'border_count': 32,
          'verbose': 200,
          'random_seed': SEED
         }
cbc_4 = ctb.CatBoostClassifier(**params)
cbc_4.fit(X_train, y_train, 
          eval_set=(X_valid, y_valid), 
          use_best_model=True, 
          plot=True
         );



np.random.seed(SEED)
noise_cols = [f'noise_{i}' for i in range(5)]
for col in noise_cols:
    X_train[col] = y_train * np.random.rand(X_train.shape[0])
    X_valid[col] = np.random.rand(X_valid.shape[0])


print(X_train.head())


params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'cat_features': cat_features,
          'verbose': 200,
          'random_seed': SEED
         }
cbc_5 = ctb.CatBoostClassifier(**params)
cbc_5.fit(X_train, y_train, 
          eval_set=(X_valid, y_valid), 
          use_best_model=True, 
          plot=True
         );


ignored_features = list(range(X_train.shape[1] - 5, X_train.shape[1]))
print(ignored_features)



params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'cat_features': cat_features,
          'ignored_features': ignored_features,
          'early_stopping_rounds': 200,
          'verbose': 200,
          'random_seed': SEED
         }
cbc_6 = ctb.CatBoostClassifier(**params)
cbc_6.fit(X_train, y_train, 
          eval_set=(X_valid, y_valid), 
          use_best_model=True, 
          plot=True
         );


X_train = X_train.drop(columns=noise_cols)
X_valid = X_valid.drop(columns=noise_cols)

print(X_train.head())


from catboost import Pool

train_data = Pool(data=X_train,
                  label=y_train,
                  cat_features=cat_features
                 )

valid_data = Pool(data=X_valid,
                  label=y_valid,
                  cat_features=cat_features
                 )


params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
#           'cat_features': cat_features, # we don't need to specify this parameter as 
#                                           pool object contains info about categorical features
          'early_stopping_rounds': 200,
          'verbose': 200,
          'random_seed': SEED
         }

cbc_7 = ctb.CatBoostClassifier(**params)
cbc_7.fit(train_data, # instead of X_train, y_train
          eval_set=valid_data, # instead of (X_valid, y_valid)
          use_best_model=True, 
          plot=True
         );



from catboost import cv

params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'verbose': 200,
          'random_seed': SEED
         }

all_train_data = Pool(data=X,
                      label=y,
                      cat_features=cat_features
                     )

scores = cv(pool=all_train_data,
            params=params, 
            fold_count=4,
            seed=SEED, 
            shuffle=True,
            stratified=True, # if True the folds are made by preserving the percentage of samples for each class
            plot=True
           )

cbc_7.get_feature_importance(prettified=True)


feature_importance_df = pd.DataFrame(cbc_7.get_feature_importance(prettified=True), columns=['Feature Id', 'Importances'])
feature_importance_df



from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6));
sns.barplot(x="Importances", y="Feature Id", data=feature_importance_df);
plt.title('CatBoost features importance:');


import shap
explainer = shap.TreeExplainer(cbc_7) # insert your model
shap_values = explainer.shap_values(train_data) # insert your train Pool object

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[:100,:], X_train.iloc[:100,:])


shap.summary_plot(shap_values, X_train)




from sklearn.model_selection import StratifiedKFold

n_fold = 4 # amount of data folds
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)

params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'verbose': 200,
          'random_seed': SEED
         }

test_data = Pool(data=X_test,
                 cat_features=cat_features)

scores = []
prediction = np.zeros(X_test.shape[0])
for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index] # train and validation data splits
    y_train, y_valid = y[train_index], y[valid_index]
    
    train_data = Pool(data=X_train, 
                      label=y_train,
                      cat_features=cat_features)
    valid_data = Pool(data=X_valid, 
                      label=y_valid,
                      cat_features=cat_features)
    
    model = ctb.CatBoostClassifier(**params)
    model.fit(train_data,
              eval_set=valid_data, 
              use_best_model=True
             )
    
    score = model.get_best_score()['validation']['AUC']
    scores.append(score)

    y_pred = model.predict_proba(test_data)[:, 1]
    prediction += y_pred

prediction /= n_fold
print('CV mean: {:.4f}, CV std: {:.4f}'.format(np.mean(scores), np.std(scores)))




sub = pd.read_csv('/media/lvision/Sabrent/kaggle/2013/amazon-employee-access-challenge/sampleSubmission.csv')
sub['Action'] = prediction
sub_name = 'catboost_submission.csv'
sub.to_csv(sub_name, index=False)

print(f'Saving submission file as: {sub_name}')


