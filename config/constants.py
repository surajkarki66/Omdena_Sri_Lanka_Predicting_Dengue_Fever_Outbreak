from darts.models import ARIMA, AutoARIMA, RandomForest, LightGBMModel, CatBoostModel, XGBModel, LinearRegressionModel, RegressionModel
OTHER_MODEL_LOADERS = {
    'models/Ampara_RandomForest.pt': RandomForest,
    'models/Anuradhapura_RandomForest.pt': RandomForest,
    'models/Batticaloa_RandomForest.pt': RandomForest,
    'models/Colombo_RegressionModel.pt': RegressionModel,
    'models/Galle_RegressionModel.pt': RegressionModel,
    'models/Gampaha_ARIMA.pt': ARIMA,
    'models/Hambantota_ARIMA.pt': ARIMA,
    'models/Jaffna_CatBoostModel.pt': CatBoostModel,
    'models/Kalutara_RandomForest.pt': RandomForest,
    'models/Kegalle_LightGBMModel.pt': LightGBMModel,
    'models/Kilinochchi_RegressionModel.pt': RegressionModel,
    'models/Kurunegala_AutoARIMA.pt': AutoARIMA,
    'models/Mannar_LinearRegressionModel.pt': LinearRegressionModel,
    'models/Matale_LinearRegressionModel.pt': LinearRegressionModel,
    'models/Matara_CatBoostModel.pt': CatBoostModel,
    'models/Monaragala_AutoARIMA.pt': AutoARIMA,
    'models/Mullaitivu_XGBModel.pt': XGBModel,
    'models/NuwaraEliya_CatBoostModel.pt': CatBoostModel,
    'models/Puttalam_RandomForest.pt': RandomForest,
    'models/Trincomalee_RandomForest.pt': RandomForest,
    'models/Vavuniya_RandomForest.pt': RandomForest
}


DISTRICT_WITH_WEATHER_FIELD = ['Ampara', 'Batticaloa', 'Colombo', 'Trincomalee']
DISTRICT_WITHOUT_SHAP_EXPLANATION = ['Badulla', 'Gampaha', 'Hambantota', 'Kandy', 'Kurunegela', 'Monaragala', 'Polonnaruwa', 'Ratnapura']
