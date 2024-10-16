import os
import yaml

# Define directories
models_dir = 'models'
config_dir = 'config'
config_file_path = os.path.join(config_dir, 'districts.yaml')

# Ensure the config directory exists
os.makedirs(config_dir, exist_ok=True)

districts = []

for model_file in os.listdir(models_dir):
    if model_file.endswith('.pkl') or model_file.endswith('.pt'):
        # Extract district name assuming naming convention '<DistrictName>_<ModelName>.pkl'
        try:
            district_name = model_file.split('_', 1)[0]  # Get the part before the first underscore
            if not district_name:
                raise ValueError("District name is empty.")
        except Exception as e:
            print(f"Error parsing filename '{model_file}': {e}")
            continue

        districts.append({
            'name': district_name,
            'model_file': os.path.join(models_dir, model_file)
        })

# Sort the districts list by the 'name' key
districts = sorted(districts, key=lambda x: x['name'])

config = {'districts': districts}

with open(config_file_path, 'w') as file:
    yaml.dump(config, file, sort_keys=False)

print(f"Configuration file generated at {config_file_path}")
