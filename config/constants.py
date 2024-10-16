from darts.models import ARIMA, AutoARIMA, RandomForest, LightGBMModel, CatBoostModel, XGBModel, LinearRegressionModel, RegressionModel
OTHER_MODEL_LOADERS = {
    'models/Ampara_RandomForest.pt': RandomForest,
    'models/Anuradhapura_RandomForest.pt': RandomForest,
    'models/Batticaloa_LinearRegressionModel.pt': LinearRegressionModel,
    'models/Colombo_RandomForest.pt': RandomForest,
    'models/Galle_RandomForest.pt': RandomForest,
    'models/Gampaha_ARIMA.pt': ARIMA,
    'models/Hambantota_ARIMA.pt': ARIMA,
    'models/Jaffna_RandomForest.pt': RandomForest,
    'models/Kalutara_RandomForest.pt': RandomForest,
    'models/Kegalle_LightGBMModel.pt': LightGBMModel,
    'models/Kilinochchi_RegressionModel.pt': RegressionModel,
    'models/Kurunegala_AutoARIMA.pt': AutoARIMA,
    'models/Mannar_LinearRegressionModel.pt': LinearRegressionModel,
    'models/Matale_LightGBMModel.pt': LightGBMModel,
    'models/Matara_CatBoostModel.pt': CatBoostModel,
    'models/Monaragala_AutoARIMA.pt': AutoARIMA,
    'models/Mullaitivu_XGBModel.pt': XGBModel,
    'models/NuwaraEliya_CatBoostModel.pt': CatBoostModel,
    'models/Puttalam_RandomForest.pt': RandomForest,
    'models/Trincomalee_RandomForest.pt': RandomForest,
    'models/Vavuniya_RandomForest.pt': RandomForest
}


DISTRICT_WITH_WEATHER_FIELD = ['Ampara', 'Batticaloa', 'Colombo', 'Trincomalee', 'Jaffna']
DISTRICT_WITHOUT_SHAP_EXPLANATION = ['Badulla', 'Gampaha', 'Hambantota', 'Kandy', 'Kurunegela', 'Monaragala', 'Polonnaruwa', 'Ratnapura']
