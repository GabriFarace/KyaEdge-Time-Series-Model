import json

import numpy as np



NUMBER_OF_LESSORS = 3
START_DATE =  "2023-01-01"
flood_hazard_protective_measures = [
    "flood_barriers",
    "water_proofing",
    "retention_basins",
    "sufficient_asset_elevation"
]

landslide_hazard_protective_measures =  [
    "net_barriers",
    "retaining_walls",
    "buffer_zones",
    "large_distance_from_steep_slopes_and_or_position_away_from_low_points"
]

climatic_hazard_protective_measures =  [
    "drainage_systems",
    "windbreak_walls",
    "snow_fencing",
    "ventilation",
    "insulation",
    "weather_resistant_shelter"
]
seismic_hazard_protective_measures = [
    "anchoring",
    "position_on_stable_ground",
    "generic_shaking_restistance_measures",
    "non_structural_elements_bracing",
    "sufficient_distance_from_weak_points"
]

with open("cities_data.json", "r") as f:
    cities_data = json.load(f)

with open("categories.json", "r") as f:
    categories = json.load(f)



class AssetDataGenerator:

    def __init__(self):
        self.asset_counter = 0

    def generate_new_asset(self) -> dict:
        ''' Generate sintethic asset data : category, contract_data, esg inputs and others'''
        new_asset = {}
        id = self.asset_counter
        category = np.random.choice(categories)
        city_data = np.random.choice(cities_data)
        lessor = int(np.random.choice(np.arange(1, NUMBER_OF_LESSORS + 1)))
        contract_data = self._get_contract_data(category)
        protective_measures = self._get_protective_measures()

        new_asset["id"] = id
        new_asset["united_nations_category_id"] = category["id"]
        new_asset["ritchie_bros_category_id"] = 0
        new_asset["lessor_id"] = lessor
        new_asset["lessee_id"] = 0
        new_asset["purchase_cost"] = category["cost"]
        new_asset["applied_life"] = {
            "years" : 0,
            "units" : 0,
            "hours" : 0,
            "kwh" : 0
        }
        new_asset["residual_value"] = category["residual_value"]
        new_asset["contract_data"] = contract_data
        new_asset["esg_inputs"] = {
            "longitude" : city_data["lon"],
            "latitude" : city_data["lat"],
            "address" : {
                "street" : "",
                "city" : city_data["name"],
                "postalCode" : "",
                "country" : "Italy"
            },
            "protective_measures" : protective_measures
        }
        # Registration date (We assume asset a registered the 1 of January 2020
        new_asset["start_date"] = START_DATE  # format '%Y-%m-%d'

        new_asset["category_data"] = category
        new_asset["city_data"] = city_data

        return new_asset


    def _get_contract_data(self, category) -> dict:
        ''' Generate and return random contract data based on the asset category data'''
        # minimum 3 years
        years = int(np.random.choice(np.arange(3, category["useful_life_years"] + 1)))
        contract_months = years * 12
        contract_amount = (years/category["useful_life_years"]) * (category["cost"] - category["residual_value"])

        # Upfront payment equal to 1 up to 5 times the ratio between contract_amount and contract_months
        contract_upfront_payment = float(np.random.uniform(1, 5) * (contract_amount/contract_months))


        contract_monthly_payment = (contract_amount - contract_upfront_payment) / contract_months
        contract_redemption_value = 0 # assume 0 redemption value

        return {
            "contract_months": contract_months,
            "contract_amount": contract_amount,
            "contract_upfront_payment": contract_upfront_payment,
            "contract_monthly_payment": contract_monthly_payment,
            "contract_redemption_value": contract_redemption_value
        }

    def _get_protective_measures(self) -> dict:
        ''' Generate the protective measures used for the asset'''
        number_flood = np.random.choice(np.arange(len(flood_hazard_protective_measures) + 1))
        flood_pm = []
        if number_flood > 0:
            flood_pm = np.random.choice(flood_hazard_protective_measures, number_flood, replace=False).tolist()

        number_landslide = np.random.choice(np.arange(len(landslide_hazard_protective_measures) + 1))
        landslide_pm = []
        if number_landslide > 0:
            landslide_pm = np.random.choice(landslide_hazard_protective_measures, number_landslide, replace=False).tolist()

        number_climatic = np.random.choice(np.arange(len(climatic_hazard_protective_measures) + 1))
        climatic_pm = []
        if number_climatic > 0:
            climatic_pm = np.random.choice(climatic_hazard_protective_measures, number_climatic, replace=False).tolist()

        number_seismic = np.random.choice(np.arange(len(seismic_hazard_protective_measures) + 1))
        seismic_pm = []
        if number_seismic > 0:
            seismic_pm = np.random.choice(seismic_hazard_protective_measures, number_seismic, replace=False).tolist()

        return {
            "flood_hazard_protective_measures" : flood_pm,
            "landslide_hazard_protective_measures" : landslide_pm,
            "climatic_hazard_protective_measures" : climatic_pm,
            "seismic_hazard_protective_measures" : seismic_pm
        }





