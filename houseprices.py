import numpy as np
import pandas as pd
import pickle
from lofo import LOFOImportance, Dataset, plot_importance

pd.set_option("max_columns", 5000)
from scipy.stats import norm, skew, probplot
from scipy.special import boxcox1p
import warnings

warnings.filterwarnings(action="ignore")
import seaborn as sns

color = sns.color_palette()
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    PolynomialFeatures,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import roc_auc_score, make_scorer
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna


test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
usdtry = pd.read_csv("data/usdtry.csv")
credit_rates = pd.read_csv("data/kredi_oranlari.csv")
istanbul_index = pd.read_csv("data/istanbul_index.csv")
izmir_index = pd.read_csv("data/izmir_index.csv")
submission = pd.read_csv("data/sample_submission_zingat.csv")


izmir_index["date"] = pd.to_datetime(izmir_index["month"], format="%m-%Y")
izmir_index = izmir_index.set_index("date")
izmir_index_1 = izmir_index.resample("D").ffill()

istanbul_index["date"] = pd.to_datetime(istanbul_index["month"], format="%m-%Y")
istanbul_index = istanbul_index.set_index("date")
istanbul_index_1 = istanbul_index.resample("D").ffill()


usdtry.fillna(method="ffill", inplace=True)
usdtry['usdtry'] = usdtry.usdtry.cummax()
credit_rates.fillna(method="ffill", inplace=True)
usdtry["date"] = pd.to_datetime(usdtry.date, format="%d-%m-%Y", errors="coerce")
credit_rates["date"] = pd.to_datetime(
    credit_rates.date, format="%d-%m-%Y", errors="coerce"
)
usdtry["usdtry"] = usdtry["usdtry"].astype("float")

train["is_train"] = 1
test["is_train"] = 0

all = pd.concat([train, test], ignore_index=True)
train_idxes = all[all.is_train == 1].index
all.loc[all.place == "İzmir", "place"] = "İzmir/Karşıyaka/Bostanlı"
all[["city", "county", "district"]] = all.place.str.split(pat="/", expand=True)
all["county"] = all["city"] + "-" + all["county"]
all["district"] = all["county"] + "-" + all["district"]

all["date"] = pd.to_datetime(all.date, format="%Y-%m-%d", errors="coerce")
all = pd.merge(all, istanbul_index_1, on="date", how="left")
all = pd.merge(all, izmir_index_1, on="date", how="left")

izmir_201901_index = 3000
istanbul_201901_index = 3527.11

all["index_base"] = -1
all["index_current"] = -1
all.loc[all.city == "İzmir", "index_base"] = izmir_201901_index
all.loc[all.city == "İstanbul", "index_base"] = istanbul_201901_index
all.loc[all.city == "İzmir", "index_current"] = all.izmir_index
all.loc[all.city == "İstanbul", "index_current"] = all.istanbul_index
all["index_multiplier"] = all["index_current"] / all["index_base"]

all.loc[all.room == "-", "room"] = "3+1"
all[["nroom_count", "lroom_count"]] = all.room.str.split(pat="+", expand=True)
all["nroom_count"] = all["nroom_count"].astype("int32")
all["lroom_count"] = all["lroom_count"].astype("int32")
all["room_count"] = all["nroom_count"] + all["lroom_count"]
all["avg_room_area"] = all["net_area"] / all["room_count"]

# area corrections
all.at[4043, 'gross_area'] = 100
all.at[6334, 'gross_area'] = 150
all.at[7316, 'gross_area'] = 100
all.at[9931, 'gross_area'] = 125
all.at[20087, 'gross_area'] = 95
all.at[26265, 'gross_area'] = 145
all.at[31045, 'gross_area'] = 120
all.at[37515, 'gross_area'] = 350
all.at[42195, 'gross_area'] = 145
all.at[42941, 'gross_area'] = 80
all.at[48835, 'gross_area'] = 240
all.at[54531, 'gross_area'] = 78
all.at[64798, 'gross_area'] = 130
all.at[64859, 'gross_area'] = 130
all.at[65891, 'gross_area'] = 65
all.at[69588, 'gross_area'] = 100
all.at[72085, 'gross_area'] = 90
all.at[77161, 'gross_area'] = 270
all.at[81276, 'gross_area'] = 148
all.at[1250, 'gross_area'] = 600
all.at[5294, 'gross_area'] = 950
all.at[6491, 'gross_area'] = 300
all.at[14828, 'gross_area'] = 100
all.at[34195, 'gross_area'] = 90
all.at[34223, 'gross_area'] = 100
all.at[34305, 'gross_area'] = 120
all.at[34310, 'gross_area'] = 110
all.at[34408, 'gross_area'] = 100
all.at[34417, 'gross_area'] = 70
all.at[36094, 'gross_area'] = 125
all.at[43910, 'gross_area'] = 100
all.at[46614, 'gross_area'] = 130
all.at[55043, 'gross_area'] = 95
all.at[56082, 'gross_area'] = 1070
all.at[59788, 'gross_area'] = 90
all.at[59816, 'gross_area'] = 135
all.at[60218, 'gross_area'] = 656
all.at[60442, 'gross_area'] = 290
all.at[63206, 'gross_area'] = 250
all.at[63212, 'gross_area'] = 231
all.at[63830, 'gross_area'] = 200
all.at[64863, 'gross_area'] = 800
all.at[75791, 'gross_area'] = 315
all.at[79578, 'gross_area'] = 150
all.at[81757, 'gross_area'] = 400
all.at[82936, 'gross_area'] = 500
all.at[91386, 'gross_area'] = 250
all.at[53907, 'gross_area'] = 140
all.at[9374, 'gross_area'] = 205
all.at[9528, 'gross_area'] = 120
all.at[30298, 'gross_area'] = 55
all.at[40757, 'gross_area'] = 110
all.at[48386, 'gross_area'] = 100
all.at[50987, 'gross_area'] = 108
all.at[60052, 'gross_area'] = 130
all.at[60503, 'gross_area'] = 95
all.at[64984, 'gross_area'] = 135
all.at[65871, 'gross_area'] = 400
all.at[68884, 'gross_area'] = 130

all.at[59781, 'net_area'] = 110
all.at[59815, 'net_area'] = 100
all.at[60229, 'net_area'] = 300
all.at[72369, 'net_area'] = 220
all.at[85441, 'net_area'] = 75
all.at[6491, 'net_area'] = 120
all.at[14828, 'net_area'] = 98
all.at[34195, 'net_area'] = 70
all.at[34223, 'net_area'] = 70
all.at[34305, 'net_area'] = 110
all.at[34310, 'net_area'] = 80
all.at[34408, 'net_area'] = 80
all.at[34417, 'net_area'] = 60
all.at[36094, 'net_area'] = 120
all.at[43910, 'net_area'] = 91
all.at[46614, 'net_area'] = 115
all.at[55043, 'net_area'] = 90
all.at[59788, 'net_area'] = 60
all.at[60218, 'net_area'] = 553
all.at[60442, 'net_area'] = 145
all.at[63206, 'net_area'] = 200
all.at[63212, 'net_area'] = 171
all.at[63830, 'net_area'] = 170
all.at[64863, 'net_area'] = 180
all.at[75791, 'net_area'] = 250
all.at[79578, 'net_area'] = 130
all.at[81757, 'net_area'] = 143
all.at[82936, 'net_area'] = 424
all.at[91386, 'net_area'] = 200
all.at[78397, 'net_area'] = 130
all.at[96874, 'net_area'] = 100
all.at[97142, 'net_area'] = 100
all.at[60503, 'net_area'] = 85
all.at[70053, 'net_area'] = 130
all.at[70195, 'net_area'] = 185


all["gross_area"] = all["gross_area"].astype("int")
all["net_area"] = all["gross_area"].astype("int")
all["gross_area - net_area"] = all["gross_area"] - all["net_area"]
all["gross_area / net_area"] = all["gross_area"] / all["net_area"]
all["gross_area_squared"] = all["gross_area"] ** 2
all["gross_area_log"] = np.log1p(all["gross_area"])

all.loc[all.building_age == "6-10 arası", "building_age"] = "8"
all.loc[all.building_age == "11-15 arası", "building_age"] = "13"
all.loc[all.building_age == "16-20 arası", "building_age"] = "18"
all.loc[all.building_age == "21-25 arası", "building_age"] = "23"
all.loc[all.building_age == "26-30 arası", "building_age"] = "28"
all.loc[all.building_age == "31-35 arası", "building_age"] = "33"
all.loc[all.building_age == "36-40 arası", "building_age"] = "38"
all.loc[all.building_age == "40 ve üzeri", "building_age"] = "45"
all.loc[all.building_age == "-", "building_age"] = None
all["building_age"] = all["building_age"].astype("float")
all["building_age"] = all["building_age"].fillna(
    all.groupby("district")["building_age"].transform("mean")
)
all["building_age"].fillna(all["building_age"].mean(), inplace=True)

all["is_summerplace"] = 0
all.loc[all.building_type == "Yazlık", "is_summerplace"] = 1

all["is_new"] = 0
all.loc[all.building_age == 0.0, "is_new"] = 1
all["is_mustakil"] = 0
all.loc[all.floor_number == "Müstakil", "is_mustakil"] = 1
all.loc[all.floor_number == "Komple", "is_mustakil"] = 1
all.loc[all.building_type == "Müstakil Ev", "is_mustakil"] = 1
all["is_whole_building"] = 0
all.loc[all.floor_number == "Komple", "is_whole_building"] = 1

all["is_teras"] = 0
all.loc[all.floor_number == "Teras Kat", "is_teras"] = 1

all["is_underground"] = 0
all.loc[
    all.floor_number.isin(["Bodrum Kat", "Kot 1", "Kot 2", "Kot 3", "Kot 4"]),
    "is_underground",
] = 1

all["is_top"] = 0
all.loc[all.floor_number == "En Üst Kat", "is_top"] = 1
all.loc[all.floor_number == all.number_of_floors, "is_top"] = 1

all.loc[all.floor_number == "En Üst Kat", "floor_number"] = all.number_of_floors
all.loc[all.floor_number == "Teras Kat", "floor_number"] = all.number_of_floors

all.loc[all.floor_number == "Komple", "floor_number"] = all.number_of_floors
all.loc[all.floor_number == "Müstakil", "floor_number"] = "0"
all.loc[all.floor_number == "-", "floor_number"] = "0"
all.loc[all.floor_number == "Bahçe katı", "floor_number"] = "0"
all.loc[all.floor_number == "Yüksek Giriş", "floor_number"] = "0"
all.loc[all.floor_number == "Giriş Katı", "floor_number"] = "0"
all.loc[all.floor_number == "Zemin Kat", "floor_number"] = "0"
all.loc[all.floor_number == "Çatı Katı", "floor_number"] = "0"
all.loc[all.floor_number == "20 ve üzeri", "floor_number"] = "20"
all.loc[all.floor_number == "10-20 arası", "floor_number"] = "15"
all.loc[all.floor_number == "Bodrum Kat", "floor_number"] = "-1"
all.loc[all.floor_number == "Kot 1", "floor_number"] = "-1"
all.loc[all.floor_number == "Kot 2", "floor_number"] = "-2"
all.loc[all.floor_number == "Kot 3", "floor_number"] = "-3"
all.loc[all.floor_number == "Kot 4", "floor_number"] = "-4"
all["floor_number"] = all["floor_number"].astype("int")

all["is_daire"] = (all["building_type"] == "Daire").astype("int8")
all["is_villa"] = (all["building_type"] == "Villa").astype("int8")
all["is_rezidans"] = (all["building_type"] == "Rezidans").astype("int8")
all["is_prefabric"] = (all["building_type"] == "Prefabrik Ev").astype("int8")
all["is_ciftlik"] = (all["building_type"] == "Çiftlik Ev").astype("int8")
all["is_yali"] = (all["building_type"] == "Yalı Dairesi").astype("int8")
all["is_kosk"] = (all["building_type"] == "Köşk / Konak / Yalı").astype("int8")

all.loc[all.number_of_floors == "20 ve üzeri", "number_of_floors"] = "20"
all.loc[all.number_of_floors == "10-20 arası", "number_of_floors"] = "15"
all.loc[all.number_of_floors == "-", "number_of_floors"] = None
all["number_of_floors"] = all["number_of_floors"].astype("float")
# TODO fillna better
all.number_of_floors.fillna(all.number_of_floors.median(), inplace=True)

all.loc[all.heating == "-", "heating"] = "Yok"
all["is_heating"] = 1
all.loc[all.heating == "Yok", "is_heating"] = 0

all.loc[all.bath_count == "-", "bath_count"] = 0
all.loc[all.bath_count == "6 ve üzeri", "bath_count"] = 6
all["bath_count"] = all["bath_count"].astype("int")

all["scene_city"] = all["scene"].str.contains("Şehir").astype("int8")
all["scene_Doğa"] = all["scene"].str.contains("Doğa").astype("int8")
all["scene_Cadde"] = all["scene"].str.contains("Cadde").astype("int8")
all["scene_Deniz"] = all["scene"].str.contains("Deniz").astype("int8")
all["scene_Boğaz"] = all["scene"].str.contains("Boğaz").astype("int8")
all["scene_Dağ"] = all["scene"].str.contains("Dağ").astype("int8")
all["scene_Göl"] = all["scene"].str.contains("Göl").astype("int8")
all["scene_Havuz"] = all["scene"].str.contains("Havuz").astype("int8")
all["scene_Nehir"] = all["scene"].str.contains("Nehir").astype("int8")
all["scene_Park"] = all["scene"].str.contains("Park").astype("int8")
all["scene_Vadi"] = all["scene"].str.contains("Vadi").astype("int8")
all["scene_Yeşil"] = all["scene"].str.contains("Yeşil Alan").astype("int8")

all.loc[all.carpark == "-", "carpark"] = "Yok"
all["carpark_kapali"] = all["carpark"].str.contains("Kapalı").astype("int8")
all["carpark_acik"] = all["carpark"].str.contains("Açık").astype("int8")
all["carpark_Ücretli"] = all["carpark"].str.contains("Ücretli").astype("int8")
all["price"] = all["price"].str.strip("TRY").astype("float")

all["normalized_price"] = all["price"] / all["index_multiplier"]

all["district_priceperare"] = all.iloc[train_idxes].groupby("district")["normalized_price"].transform('mean') / all.iloc[train_idxes].groupby("district")["net_area"].transform('mean')
all["city_priceperare"] = all.iloc[train_idxes].groupby("city")["normalized_price"].transform('mean') / all.iloc[train_idxes].groupby("city")["net_area"].transform('mean')
all["county_priceperare"] = all.iloc[train_idxes].groupby("county")["normalized_price"].transform('mean') / all.iloc[train_idxes].groupby("county")["net_area"].transform('mean')

all = pd.merge(all, usdtry, on="date", how="left")
all["price_in_usd"] = all["price"] / all["usdtry"]
all["price_in_usd_per_meter"] = all["price_in_usd"] / all["gross_area"]
all["price_in_usd_log"] = np.log1p(all["price_in_usd"])

basedate = pd.Timestamp("2019-01-01")
all["days_passed"] = (all["date"] - basedate).dt.days

all = pd.merge(all, credit_rates, on="date", how="left")


all["floor_number"] = all["floor_number"] + 4

all["priceperarea"] = all["price"] / all["net_area"]

adversarial_validation = False
run_train = True
run_plot = False
lofo_importance = False
check_val_results = True
trial = 38

features = [
    "building_type",
    "building_age",
    "floor_number",
    "number_of_floors",
    "heating",
    "bath_count",
    "carpark",
    "earthquake_regulations",
    "elevator",
    "playground",
    "parent_bath",
    "city",
    "county",
    "district",
    "nroom_count",
    "lroom_count",
    "room_count",
    "avg_room_area",
    "is_summerplace",
    "scene_city",
    "scene_Doğa",
    "scene_Cadde",
    "scene_Dağ",
    "scene_Havuz",
    "scene_Vadi",
    "scene_Yeşil",
    "is_new",
    "is_mustakil",
    "is_whole_building",
    "net_area",
    "is_underground",
    "is_top",
    "is_teras",
    "gross_area - net_area",
    "gross_area / net_area",
    "gross_area",
    "is_heating",
    "carpark_Ücretli",
    "scene_Park",
    "scene_Göl",
    "changing_room",
    "scene_Deniz",
    "scene_Boğaz",
    "scene_Nehir",
    "carpark_acik",
    "is_daire",
    "is_villa",
    "is_rezidans",
    "is_prefabric",
    "is_ciftlik",
    "is_yali",
    "is_kosk",
    "credit_rate",
    "carpark_kapali",
    "intercom",
]
categorical_features = [
    #"building_type",
    "heating",
    "carpark",
    "intercom",
    "earthquake_regulations",
    "elevator",
    "playground",
    "changing_room",
    "parent_bath",
    "changing_room",
    "city",
    "county",
    "district",
    "is_summerplace",
    "is_new",
    "is_mustakil",
    "is_whole_building",
    "is_teras",
    "is_underground",
    "is_top",
    "is_heating",
    "scene_city",
    "scene_Doğa",
    "scene_Cadde",
    "scene_Deniz",
    "scene_Boğaz",
    "scene_Dağ",
    "scene_Göl",
    "scene_Havuz",
    "scene_Nehir",
    "scene_Park",
    "scene_Vadi",
    "scene_Yeşil",
    "carpark_kapali",
    "carpark_acik",
    "carpark_Ücretli",
    "is_daire",
    "is_villa",
    "is_rezidans",
    "is_prefabric",
    "is_ciftlik",
    "is_yali",
    "is_kosk",
]
target_feature = "price_in_usd_log"

all[categorical_features] = all[categorical_features].astype("category")
le = LabelEncoder()
all[categorical_features] = all[categorical_features].apply(le.fit_transform)


if adversarial_validation:
    adv_ban_list = ["days_passed", "credit_rate", "building_type"]
    features_adv = [feat for feat in features if feat not in adv_ban_list]
    y = all["is_train"]
    X = all[features_adv]

    folds = KFold(n_splits=5, shuffle=True, random_state=15)
    oof = np.zeros(len(X))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values, y)):
        print("fold n°{}".format(fold_))

        classifier = RFC(n_estimators=1000, n_jobs=-1, verbose=True)
        classifier.fit(all.iloc[trn_idx][features_adv], y.iloc[trn_idx])
        oof[val_idx] = classifier.predict_proba(all.iloc[val_idx][features_adv])[:, 1]
    print(f"ROC Adv score: {roc_auc_score(y, oof)}")
    train_idxes = all[all.is_train == 1].index
    val_idxes = oof[train_idxes].argsort()[:5000]
    train_idx_to_drop = oof[train_idxes].argsort()[-5000:]
    train_train_idxes = train_idxes.difference(val_idxes).difference(train_idx_to_drop)
    with open("validation_files/val_indexes.pickle", "wb") as handle:
        pickle.dump(val_idxes, handle)
    with open("validation_files/train_indexes.pickle", "wb") as handle:
        pickle.dump(train_train_idxes, handle)
else:
    with open("validation_files/val_indexes.pickle", "rb") as handle:
        val_idxes = pickle.load(handle)
    with open("validation_files/train_indexes.pickle", "rb") as handle:
        train_train_idxes = pickle.load(handle)



features.remove("building_type")
train_x = all.iloc[train_train_idxes][features]
train_y = all.iloc[train_train_idxes][target_feature]

additional_idxes = train_y[train_y > 14].index
train_x = pd.concat([train_x, train_x.loc[additional_idxes]], axis=0)
train_y = pd.concat([train_y, train_y.loc[additional_idxes]], axis=0)

val_x = all.iloc[val_idxes][features]
val_y = all.iloc[val_idxes][target_feature]

X_test = all[all.is_train == 0][features]


if lofo_importance:
    # extract a sample of the data
    sample_df = all.iloc[train_idxes].sample(frac=0.01, random_state=0)
    # define the validation scheme
    cv = KFold(n_splits=4, shuffle=False, random_state=0)

    # define the binary target and the features
    dataset = Dataset(df=sample_df, target=target_feature, features=features)

    # define the validation scheme and scorer. The default model is LightGBM
    lofo_imp = LOFOImportance(dataset, cv=cv, scoring="neg_mean_absolute_error")

    # get the mean and standard deviation of the importances in pandas format
    importance_df = lofo_imp.get_importance()
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        print(importance_df)

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        print(importance_df)

    # plot the means and standard deviations of the importances
    plot_importance(importance_df, figsize=(12, 20))

    import_features = importance_df[
        importance_df.importance_mean > 0
    ].feature.values.tolist()
    print(import_features)

if run_plot:
    sns.distplot(train_y, fit=norm)
    plt.show()

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train_y)
    print("\n mu = {:.2f} and sigma = {:.2f}\n".format(mu, sigma))


def objective(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param = {
        "tree_method": "gpu_hist",  # Use GPU acceleration
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 1e3),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 1e3),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 0.9, 1.0]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
        "learning_rate": trial.suggest_float("eta", 0.002, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 400, 4000, 400),
        "max_depth": trial.suggest_int("max_depth", 6, 15),
        "random_state": 42,
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-4, 1e4),
        "objective": 'reg:squarederror'
    }
    model = XGBRegressor(**param)

    model.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        eval_metric="rmse",
        early_stopping_rounds=100,
        verbose=False,
    )

    preds = model.predict(val_x)

    rmse = mse(val_y, preds, squared=False)

    return rmse


if run_train:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=60)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)

    best_params = study.best_params
    # best_params = {'lambda': 8.288615656671075, 'alpha': 0.012066435823310209, 'colsample_bytree': 0.8, 'subsample': 0.8, 'eta': 0.00761510354047123, 'n_estimators': 4000, 'max_depth': 13, 'min_child_weight': 0.0009035921654047102}
    best_params["tree_method"] = "gpu_hist"
    best_params["random_state"] = 42

    clf = XGBRegressor(**(best_params))

    clf.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        eval_metric="rmse",
        early_stopping_rounds=200,
    )
    clf.save_model(f"models/{trial}.model")

if check_val_results:
    model = XGBRegressor({"nthread": 4})  # init model
    model.load_model(f"models/{trial}.model")  # load data
    preds = model.predict(val_x)
    val_x["preds"] = preds
    val_x["preds"] = np.exp(val_x["preds"]) * all.iloc[val_idxes]["usdtry"]
    val_x["price_target"] = np.exp(val_y) * all.iloc[val_idxes]["usdtry"]
    val_x["rmse"] = ((val_x.preds - val_x.price_target) ** 2) ** 0.5
    val_score = ((val_x.preds - val_x.price_target) ** 2).mean() ** 0.5
    print(f"Validation score: {val_score}")

model = XGBRegressor({"nthread": 4})  # init model
model.load_model(f"models/{trial}.model")  # load data
preds = model.predict(X_test)
submission["Expected"] = preds
submission = pd.merge(submission, all, left_on="Id", right_on="id")
submission["Expected"] = np.exp(submission["Expected"].abs()) * submission["usdtry"]
submission[["Id", "Expected"]].to_csv(
    f"submission_files/submission_{trial}.csv", sep=",", index=False
)
