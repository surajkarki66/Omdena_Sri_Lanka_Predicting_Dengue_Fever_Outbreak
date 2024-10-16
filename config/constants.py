from darts.models import TransformerModel, NBEATSModel

MODEL_LOADERS = {
    'models/Badulla_TransformerModel.pt': TransformerModel,
    'models/Kandy_TransformerModel.pt': TransformerModel,
    'models/Jaffna_NBEATSModel.pt': NBEATSModel,
    'models/Polonnaruwa_TransformerModel.pt': TransformerModel,
    'models/Ratnapura_TransformerModel.pt': TransformerModel,
}

DISTRICT_WITH_WEATHER_FIELD = ['Ampara', 'Batticaloa', 'Colombo', 'Trincomalee', 'Jaffna']
DISTRICT_WITHOUT_SHAP_EXPLANATION = ['Badulla', 'Gampaha', 'Hambantota', 'Kandy', 'Kurunegela', 'Monaragala', 'Polonnaruwa', 'Ratnapura']
