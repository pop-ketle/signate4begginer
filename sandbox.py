import csv
import time
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
import category_encoders as ce
from optuna.integration import lightgbm_tuner

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier 

np.set_printoptions(suppress=True)

train_x = pd.read_csv('./datasets/train.csv')
test_x  = pd.read_csv('./datasets/test.csv')

train_y = train_x['y']
train_x = train_x.drop(['y'], axis=1)
print(len(train_x), len(test_x))

# ラベルエンコーディングとワンホットエンコーディングと標準化
n_idx = len(train_x) # あとでtrainとtestを分離するための、長さ
features = pd.concat([train_x, test_x]) # 縦方向に連結
ce_oe = ce.OrdinalEncoder(cols=['job','marital','education','default','housing','loan','contact','month','poutcome'], handle_unknown='impute')

# ワンホットエンコーディングと標準化
scaler = StandardScaler()
features_onehot = pd.get_dummies(features, columns=['job','marital','education','default','housing','loan','contact','month','poutcome'])
train_onehot_x, test_onehot_x = features_onehot[:n_idx], features_onehot[n_idx:] # データの分離
train_std_x, test_std_x = scaler.fit_transform(train_onehot_x), scaler.fit_transform(test_onehot_x)

# ラベルエンコーディング
features = ce_oe.fit_transform(features)
train_x, test_x = features[:n_idx], features[n_idx:] # データの分離

# パラメーターチューニング
x_train,x_valid,y_train,y_valid = train_test_split(train_x, train_y, random_state=0, test_size=0.3)

gbm_params = {"objective": 'binary', "metric": 'auc'}
train_data, valid_data = lgb.Dataset(x_train, y_train), lgb.Dataset(x_valid, y_valid)

# # Optuna でハイパーパラメータを Stepwise Optimization する
# best_params = {}
# gbm = lightgbm_tuner.train(gbm_params, train_data,
#                                         valid_sets=valid_data,
#                                         num_boost_round=1000,
#                                         early_stopping_rounds=50,
#                                         verbose_eval=10,
#                                         best_params=best_params,
#                                         )
# gbm_params.update(dict(best_params))
# print(f'gbm_params{best_params}, best_score:{gbm.best_score}')
# # gbm_params{'lambda_l1': 2.3820381694990465, 'lambda_l2': 0.4671166147125861, 'num_leaves': 3, 'feature_fraction': 0.6, 'bagging_fraction': 0.8904710948966893, 'bagging_freq': 3, 'min_child_samples': 20}, best_score:defaultdict(<class 'dict'>, {'valid_0': {'auc': 0.8533417542933542}})

# def objective(trial):
#     n_estimators = trial.suggest_int("n_estimators", 10, 1000)
#     min_samples_split = trial.suggest_int("min_samples_split", 8, 64)
#     max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 4, 128, 4)
#     criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

#     RFC = RandomForestClassifier(
#                         random_state = 777,
#                         n_estimators = n_estimators,
#                         min_samples_split = min_samples_split, 
#                         max_leaf_nodes = max_leaf_nodes,
#                         criterion = criterion)
#     RFC.fit(x_train, y_train)
#     rfc_valid_pred  = RFC.predict_proba(x_valid)
#     idx = 0 if RFC.classes_[0]==0 else 1
#     rfc_valid_pred = 1 - rfc_valid_pred[:,idx]
#     return roc_auc_score(y_valid, rfc_valid_pred)

# study = optuna.create_study()
# study.optimize(objective, n_trials=100)
# print(f'rfc_params:{study.best_params}, best_score:{study.best_value}')
# # rfc_params:{'n_estimators': 11, 'min_samples_split': 34, 'max_leaf_nodes': 4.0, 'criterion': 'entropy'}, best_score:0.5
# # rfc_params:{'n_estimators': 94, 'min_samples_split': 61, 'max_leaf_nodes': 4, 'criterion': 'entropy'}, best_score:0.7668523003562523

# x_train,x_valid,y_train,y_valid = train_test_split(train_std_x, train_y, random_state=0, test_size=0.3)
# print(len(x_train), len(x_valid), len(y_train),len(y_valid))
# def objective(trial):
#     params = {
#         "C": trial.suggest_uniform("C", 0.1, 10),
#         "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
#         "intercept_scaling": trial.suggest_uniform("intercept_scaling", 0.1, 2),
#         "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "saga"]),
#         "max_iter": trial.suggest_int("max_iter", 100, 1000),
#     }
#     LR = LogisticRegression(**params)
#     LR.fit(x_train, y_train)
#     lr_valid_pred  = LR.predict_proba(x_valid)
#     idx = 0 if LR.classes_[0]==0 else 1
#     lr_valid_pred = 1 - lr_valid_pred[:,idx]
#     return roc_auc_score(y_valid, lr_valid_pred)
# study = optuna.create_study()
# study.optimize(objective, n_trials=100)
# print(f'lr_params:{study.best_params}, best_score:{study.best_value}')
# # lr_params:{'C': 9.939520501657233, 'fit_intercept': True, 'intercept_scaling': 1.1325167020454558, 'solver': 'saga', 'max_iter': 888}, best_score:0.5961669370037228
# # lr_params:{'C': 0.5272498342894304, 'fit_intercept': True, 'intercept_scaling': 0.21350734718547665, 'solver': 'lbfgs', 'max_iter': 943}, best_score:0.7975427943896476

############## パラメータチューニングおしまい

gbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'lambda_l1': 2.3820381694990465,
    'lambda_l2': 0.4671166147125861,
    'num_leaves': 3,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8904710948966893,
    'bagging_freq': 3,
    'min_child_samples': 20
}

rfc_params = {
    'random_state': 777,
    'n_estimators': 94,
    'min_samples_split': 61,
    'max_leaf_nodes': 4,
    'criterion': 'entropy'
}

lr_params = {
    'C': 0.5272498342894304,
    'fit_intercept': True,
    'intercept_scaling': 0.21350734718547665,
    'solver': 'lbfgs',
    'max_iter': 94
}


test_x_copy = test_x[:]
lv1scores, lv2scores, preds = [], [], []
for train_idx, valid_idx in KFold(n_splits=5).split(train_x, train_y):
    x_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
    x_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]
    # x_train_std, x_valid_std = train_std_x.iloc[train_idx], train_idx.iloc[valid_idx]
    x_train_std = [train_std_x[i] for i in train_idx]
    x_valid_std = [train_std_x[i] for i in valid_idx]
    print(len(x_train), len(y_train), len(x_valid), len(y_valid))
    # test_x = test_x_copy
    # # for i in range(10):
    # #     if i!=0: # 擬似ラベルを使ってrefinning
    # #         valid_preds = np.stack([gbm_valid_pred, rfc_valid_pred, lr_valid_pred], 1)
    # #         train_preds = np.stack([gbm_train_pred, rfc_train_pred, lr_train_pred], 1)
    # #         test_preds  = np.stack([gbm_test_pred, rfc_test_pred, lr_test_pred], 1)
    # #         x_train     = np.concatenate([x_train, train_preds], 1)
    # #         x_valid     = np.concatenate([x_valid, valid_preds], 1)
    # #         test_x      = np.concatenate([test_x, test_preds], 1)
    # #     print(i,x_train.shape, x_valid.shape, test_x.shape)
    # #     print('')
            
    train_data, valid_data = lgb.Dataset(x_train, y_train), lgb.Dataset(x_valid, y_valid)

    GBM = lgb.train(gbm_params, train_data,
        valid_sets=valid_data,
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose_eval=20,
    )
    gbm_valid_pred  = GBM.predict(x_valid)
    gbm_train_pred  = GBM.predict(x_train)
    gbm_valid_score = roc_auc_score(y_valid, gbm_valid_pred)
    gbm_test_pred   = GBM.predict(test_x)

    RFC = RandomForestClassifier(**rfc_params)
    RFC.fit(x_train, y_train)
    rfc_valid_pred  = RFC.predict_proba(x_valid)
    idx = 0 if RFC.classes_[0]==0 else 1
    rfc_valid_pred = 1 - rfc_valid_pred[:,idx]
    rfc_train_pred  = RFC.predict_proba(x_train)
    rfc_train_pred = 1 - rfc_train_pred[:,idx]
    rfc_valid_score = roc_auc_score(y_valid, rfc_valid_pred)
    rfc_test_pred   = RFC.predict_proba(test_x)
    rfc_test_pred = 1 - rfc_test_pred[:,idx]

    LR = LogisticRegression(**lr_params)
    LR.fit(x_train_std, y_train)
    lr_valid_pred  = LR.predict_proba(x_valid_std)
    idx = 0 if LR.classes_[0]==0 else 1
    lr_valid_pred = 1 - lr_valid_pred[:,idx]
    lr_train_pred  = LR.predict_proba(x_train_std)
    lr_train_pred = 1 - lr_train_pred[:,idx]
    lr_valid_score = roc_auc_score(y_valid, lr_valid_pred)
    lr_test_pred   = LR.predict_proba(test_std_x)
    lr_test_pred = 1 - lr_test_pred[:,idx]

    lv1scores.append(f'gbm_score:{gbm_valid_score} rfc_score:{rfc_valid_score} lr_score:{lr_valid_score}')
    
    # lv2
    valid_pred = np.stack([gbm_valid_pred, rfc_valid_pred, lr_valid_pred], 1)
    test_pred  = np.stack([gbm_test_pred, rfc_test_pred, lr_test_pred], 1)

    x_train2, x_valid2 ,y_train2 ,y_valid2 = train_test_split(valid_pred, y_valid, random_state=0, test_size=0.3, shuffle=False)
    train_data2, valid_data2 = lgb.Dataset(x_train2, y_train2), lgb.Dataset(x_valid2, y_valid2)

    GBM = lgb.train(gbm_params, train_data2,
        valid_sets=valid_data2,
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose_eval=20,
    )
    gbm_valid_pred  = GBM.predict(x_valid2)
    gbm_valid_score = roc_auc_score(y_valid2, gbm_valid_pred)
    pred = GBM.predict(test_pred)
    lv2scores.append(gbm_valid_score)
    preds.append(pred)

for score in lv1scores: print(score)
print('')
for score in lv2scores: print(score)

preds = np.array(preds)
preds = preds.flatten()
print(preds.shape)

exit()

# x_train,x_valid,y_train,y_valid = train_test_split(train_x, train_y, random_state=0, test_size=0.3)


preds = np.mean(preds, axis=0)
preds = pd.Series(preds, index=[i for i in range(preds.shape[0])])
preds.to_csv('./output.csv')

# gbm_score:0.8425176217058927 rfc_score:0.7601418818118262 lr_score:0.8062525246505222
# gbm_score:0.8648825960644306 rfc_score:0.7656994239776289 lr_score:0.813724544191199
# gbm_score:0.8548774960221055 rfc_score:0.7562610352197751 lr_score:0.7963422335758948
# gbm_score:0.8496984446592126 rfc_score:0.7653765244928948 lr_score:0.8050052780483457
# gbm_score:0.8635475123936922 rfc_score:0.7985879205248267 lr_score:0.8165681399360943

# 0.8574294001992576
# 0.836496993951355
# 0.8510002817695125
# 0.8520776874435411
# 0.8707751022761594

# 0 gbm_score:0.8425176217058927 rfc_score:0.7718971749456284 lr_score:0.8062586736822388
# 1 gbm_score:0.8449107775497906 rfc_score:0.8139085421402604 lr_score:0.8062586736822388
# 2 gbm_score:0.8411759502855515 rfc_score:0.8199833124739258 lr_score:0.8062586736822388
# 3 gbm_score:0.8376806987570441 rfc_score:0.8275388169452176 lr_score:0.8062586736822388
# 4 gbm_score:0.8364669744966545 rfc_score:0.8251309980256879 lr_score:0.8062586736822388
# 5 gbm_score:0.8339390129574289 rfc_score:0.8242722620963278 lr_score:0.8062586736822388
# 6 gbm_score:0.8254687217676668 rfc_score:0.8239045026994251 lr_score:0.8062586736822388
# 7 gbm_score:0.8243406109488713 rfc_score:0.8322820854109966 lr_score:0.8062586736822388
# 8 gbm_score:0.8157191954796101 rfc_score:0.8223393376263035 lr_score:0.8062586736822388
# 9 gbm_score:0.8140078726526071 rfc_score:0.8297221962070881 lr_score:0.8062586736822388
# 0 gbm_score:0.8648825960644306 rfc_score:0.7746604551986279 lr_score:0.8137288012131567
# 1 gbm_score:0.8668635302820892 rfc_score:0.8561987442731229 lr_score:0.8137288012131567
# 2 gbm_score:0.8666984524306177 rfc_score:0.8622318903920906 lr_score:0.8137288012131567
# 3 gbm_score:0.8660464185674269 rfc_score:0.8654021892444922 lr_score:0.8137288012131567
# 4 gbm_score:0.8663295105276151 rfc_score:0.865255085485731 lr_score:0.8137288012131567
# 5 gbm_score:0.8652449159332765 rfc_score:0.8654334074055154 lr_score:0.8137288012131567
# 6 gbm_score:0.8655691591057226 rfc_score:0.8655228048666275 lr_score:0.8137288012131567
# 7 gbm_score:0.8654466514738284 rfc_score:0.8655296634020038 lr_score:0.8137288012131567
# 8 gbm_score:0.865572233621581 rfc_score:0.8651950141758831 lr_score:0.8137288012131567
# 9 gbm_score:0.8657110598376467 rfc_score:0.8660036118466299 lr_score:0.8137288012131567
# 0 gbm_score:0.8548774960221055 rfc_score:0.762986237148888 lr_score:0.7963550349049573
# 1 gbm_score:0.8572594914743148 rfc_score:0.8347912624765546 lr_score:0.7963550349049573
# 2 gbm_score:0.8569899523790558 rfc_score:0.8450731003301795 lr_score:0.7963550349049573
# 3 gbm_score:0.8579076180235127 rfc_score:0.8529885888004489 lr_score:0.7963550349049573
# 4 gbm_score:0.8581717047008377 rfc_score:0.8545105245889824 lr_score:0.7963550349049573
# 5 gbm_score:0.8580256747248662 rfc_score:0.8566554583918875 lr_score:0.7963550349049573
# 6 gbm_score:0.8583485526912187 rfc_score:0.8546852390245198 lr_score:0.7963550349049573
# 7 gbm_score:0.8590642418104683 rfc_score:0.8589881450210415 lr_score:0.7963550349049573
# 8 gbm_score:0.858421330617555 rfc_score:0.8555097394408 lr_score:0.7963550349049573
# 9 gbm_score:0.8583217647248472 rfc_score:0.8552496828115131 lr_score:0.7963550349049573
# 0 gbm_score:0.8496984446592126 rfc_score:0.7684041704643638 lr_score:0.8050128655392288
# 1 gbm_score:0.8481368441918231 rfc_score:0.8216565010096105 lr_score:0.8050128655392288
# 2 gbm_score:0.8484434262453205 rfc_score:0.8340523271308757 lr_score:0.8050128655392288
# 3 gbm_score:0.8487087513171411 rfc_score:0.8418975555949684 lr_score:0.8050128655392288
# 4 gbm_score:0.8486345361719401 rfc_score:0.8428068689554964 lr_score:0.8050128655392288
# 5 gbm_score:0.8476092764663536 rfc_score:0.8436391218617426 lr_score:0.8050128655392288
# 6 gbm_score:0.8475483394301984 rfc_score:0.8448353372212901 lr_score:0.8050128655392288
# 7 gbm_score:0.8481555758099409 rfc_score:0.8453394311468398 lr_score:0.8050128655392288
# 8 gbm_score:0.8467834255055403 rfc_score:0.8421320564850759 lr_score:0.8050128655392288
# 9 gbm_score:0.8471267594680031 rfc_score:0.8429775875003676 lr_score:0.8050128655392288
# 0 gbm_score:0.8635475123936922 rfc_score:0.7696364264056065 lr_score:0.8166103453541319
# 1 gbm_score:0.8619240264537869 rfc_score:0.8507346113829435 lr_score:0.8166103453541319
# 2 gbm_score:0.8605580409857289 rfc_score:0.8600297619329892 lr_score:0.8166103453541319
# 3 gbm_score:0.8607586352759523 rfc_score:0.8646484098990199 lr_score:0.8166103453541319
# 4 gbm_score:0.8594867821166633 rfc_score:0.8631989620312471 lr_score:0.8166103453541319
# 5 gbm_score:0.8577260100135911 rfc_score:0.8624769648518968 lr_score:0.8166103453541319
# 6 gbm_score:0.8594481333349773 rfc_score:0.8625044695063484 lr_score:0.8166103453541319
# 7 gbm_score:0.8584264302657423 rfc_score:0.8603576838045957 lr_score:0.8166103453541319
# 8 gbm_score:0.8566877093080493 rfc_score:0.8614038091101106 lr_score:0.8166103453541319
# 9 gbm_score:0.8557473346567182 rfc_score:0.8612800381650791 lr_score:0.8166103453541319

# 0.8526441311573694
# 0.8431692341428649
# 0.8506872045333584
# 0.8450722673893405
# 0.8755058894593778

# 0 gbm_score:0.8408905934464688 rfc_score:0.7734682425577858 lr_score:0.8048999789883413
# 1 gbm_score:0.8428887069104286 rfc_score:0.8373752777525649 lr_score:0.8048999789883413
# 2 gbm_score:0.8423260558107768 rfc_score:0.8428430729977119 lr_score:0.8048999789883413
# 3 gbm_score:0.8420542582207486 rfc_score:0.842554225374361 lr_score:0.8048999789883413
# 4 gbm_score:0.841271962574174 rfc_score:0.8427828964095138 lr_score:0.8048999789883413
# 5 gbm_score:0.8420479898261449 rfc_score:0.8438507801142453 lr_score:0.8048999789883413
# 6 gbm_score:0.8406147840838942 rfc_score:0.8432119053362092 lr_score:0.8048999789883413
# 7 gbm_score:0.839404983925329 rfc_score:0.841836619560099 lr_score:0.8048999789883413
# 8 gbm_score:0.839128422355402 rfc_score:0.8412351044139026 lr_score:0.8048999789883413
# 9 gbm_score:0.8387555782443581 rfc_score:0.8416801604307842 lr_score:0.8048999789883413
# 0 gbm_score:0.8661262274508976 rfc_score:0.7750537271032776 lr_score:0.8135550709408976
# 1 gbm_score:0.8675079072567232 rfc_score:0.8538437980177735 lr_score:0.8135550709408976
# 2 gbm_score:0.8669741564272753 rfc_score:0.86144347697523 lr_score:0.8135550709408976
# 3 gbm_score:0.8663869425097993 rfc_score:0.8642226775018524 lr_score:0.8135550709408976
# 4 gbm_score:0.866248334503801 rfc_score:0.8649566398955203 lr_score:0.8135550709408976
# 5 gbm_score:0.866451626245932 rfc_score:0.8658699126461545 lr_score:0.8135550709408976
# 6 gbm_score:0.8662019118224268 rfc_score:0.8659356964458268 lr_score:0.8135550709408976
# 7 gbm_score:0.8660888252905489 rfc_score:0.8658646323411638 lr_score:0.8135550709408976
# 8 gbm_score:0.8650402447245353 rfc_score:0.8664162041999547 lr_score:0.8135550709408976
# 9 gbm_score:0.8644939531707351 rfc_score:0.8641892355702465 lr_score:0.8135550709408976
# 0 gbm_score:0.8535744589287432 rfc_score:0.766110400639376 lr_score:0.798386337343907
# 1 gbm_score:0.8541528929753075 rfc_score:0.8395261476097329 lr_score:0.798386337343907
# 2 gbm_score:0.8543024417186834 rfc_score:0.8469657706590732 lr_score:0.798386337343907
# 3 gbm_score:0.8533534293015024 rfc_score:0.8504858992368378 lr_score:0.798386337343907
# 4 gbm_score:0.8521080029802405 rfc_score:0.8504717494373015 lr_score:0.798386337343907
# 5 gbm_score:0.8518450118785128 rfc_score:0.8516120281102846 lr_score:0.798386337343907
# 6 gbm_score:0.8514383271228725 rfc_score:0.8503685534855103 lr_score:0.798386337343907
# 7 gbm_score:0.8514136869547143 rfc_score:0.8516298373407354 lr_score:0.798386337343907
# 8 gbm_score:0.8513390345640568 rfc_score:0.850901366626673 lr_score:0.798386337343907
# 9 gbm_score:0.851669115232552 rfc_score:0.8505503052209344 lr_score:0.798386337343907
# 0 gbm_score:0.8511022154847064 rfc_score:0.7651821103162948 lr_score:0.8019584853222165
# 1 gbm_score:0.8526492451876766 rfc_score:0.8289569910938461 lr_score:0.8019584853222165
# 2 gbm_score:0.8501542588469373 rfc_score:0.8423660622641211 lr_score:0.8019584853222165
# 3 gbm_score:0.8511397192350814 rfc_score:0.8482159071170274 lr_score:0.8019584853222165
# 4 gbm_score:0.8504925328059123 rfc_score:0.8493138787562967 lr_score:0.8019584853222165
# 5 gbm_score:0.8502682505092612 rfc_score:0.850446146588343 lr_score:0.8019584853222165
# 6 gbm_score:0.849433792063417 rfc_score:0.8502312402292861 lr_score:0.8019584853222165
# 7 gbm_score:0.8489371141061475 rfc_score:0.849716303867229 lr_score:0.8019584853222165
# 8 gbm_score:0.8487648929366622 rfc_score:0.8506689484737946 lr_score:0.8019584853222165
# 9 gbm_score:0.8493301632794859 rfc_score:0.85130700570057 lr_score:0.8019584853222165
# 0 gbm_score:0.8620038346224037 rfc_score:0.7920801378146636 lr_score:0.8171253548564642
# 1 gbm_score:0.8599769886445932 rfc_score:0.8553713228904145 lr_score:0.8171253548564642
# 2 gbm_score:0.860752105602387 rfc_score:0.8588343417774689 lr_score:0.8171253548564642
# 3 gbm_score:0.8613581678543495 rfc_score:0.8601209227716463 lr_score:0.8171253548564642
# 4 gbm_score:0.8607086537904464 rfc_score:0.8601050809652095 lr_score:0.8171253548564642
# 5 gbm_score:0.8609901853219779 rfc_score:0.860162337779902 lr_score:0.8171253548564642
# 6 gbm_score:0.8612934427594797 rfc_score:0.859432030503172 lr_score:0.8171253548564642
# 7 gbm_score:0.8610580787781351 rfc_score:0.8602494677153036 lr_score:0.8171253548564642
# 8 gbm_score:0.8610908939486109 rfc_score:0.860238831073839 lr_score:0.8171253548564642
# 9 gbm_score:0.8602449414848934 rfc_score:0.8593143485124997 lr_score:0.8171253548564642

# 0.8606605682576769
# 0.8518327699789604
# 0.8563285180977115
# 0.8582511056408545
# 0.870066719749595

# gbm_score:0.8408905934464688 rfc_score:0.7734682425577858 lr_score:0.8048999789883413
# gbm_score:0.8661262274508976 rfc_score:0.7750537271032776 lr_score:0.8135550709408976
# gbm_score:0.8535744589287432 rfc_score:0.766110400639376 lr_score:0.798386337343907
# gbm_score:0.8511022154847064 rfc_score:0.7651821103162948 lr_score:0.8019584853222165
# gbm_score:0.8620038346224037 rfc_score:0.7920801378146636 lr_score:0.8171253548564642

# 0.8535547396528705
# 0.8446758700375953
# 0.8521701022561272
# 0.8578234521028036
# 0.86970429148019