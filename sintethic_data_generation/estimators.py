from datetime import datetime
import pandas as pd
import numpy as np

def days_between_dates(start_date_str, end_date_str):
    # Convert the start date string to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # Calculate the end date by adding 'number_of_months' to the start date
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Calculate the difference in days
    delta = end_date - start_date

    return delta.days

class LeasingRiskScoresEstimator:

    @staticmethod
    def get_leasing_risk_scores(asset_data, telemetry_data, number_of_units):
        remarketing_value_curve = LeasingRiskScoresEstimator._get_remarketing_value_curve(asset_data, telemetry_data)
        residual_debt = LeasingRiskScoresEstimator._get_residual_debt(asset_data, telemetry_data)
        return {
            "remarketing_value_curve" : remarketing_value_curve,
            "residual_debt" : residual_debt,
            "gap_curve" : LeasingRiskScoresEstimator._get_gap_curve(remarketing_value_curve, residual_debt)
        }

    @staticmethod
    def _get_residual_debt(asset_data, telemetry_data):
        residual_debt = []
        current_debt = asset_data["contract_data"]["contract_amount"] - asset_data["contract_data"]["contract_upfront_payment"]
        daily_subtraction = current_debt / len(telemetry_data)
        for i in range(len(telemetry_data["mean_curve"])):
            residual_debt.append(current_debt)
            current_debt = current_debt - daily_subtraction

        return {"curve" : residual_debt}

    @staticmethod
    def _get_remarketing_value_curve(asset_data, telemetry_data):
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        applied_life_lower = asset_data["applied_life"]["hours"]
        applied_life_upper = asset_data["applied_life"]["hours"]
        applied_life_mean = asset_data["applied_life"]["hours"]
        cost = asset_data["purchase_cost"]
        useful_life = asset_data["category_data"]["useful_life_hours"]
        residual_value = asset_data["residual_value"]
        for i in range(len(telemetry_data["mean_curve"])):
            lower_bound_curve.append(((cost - residual_value) - (cost - residual_value) * (applied_life_lower/useful_life)) + residual_value)
            upper_bound_curve.append(((cost - residual_value) - (cost - residual_value) * (applied_life_upper/useful_life)) + residual_value)
            mean_curve.append(((cost - residual_value) - (cost - residual_value) * (applied_life_mean/useful_life)) + residual_value)
            applied_life_mean += telemetry_data["mean_curve"][i]
            applied_life_lower += telemetry_data["lower_bound_curve"][i]
            applied_life_upper += telemetry_data["upper_bound_curve"][i]

        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }

    @staticmethod
    def _get_gap_curve(remarketing_value_curve, residual_debt):
        return {
            "lower_bound_curve" : remarketing_value_curve["lower_bound_curve"] - residual_debt["curve"],
            "mean_curve" : remarketing_value_curve["mean_curve"] - residual_debt["curve"],
            "upper_bound_curve" : remarketing_value_curve["upper_bound_curve"] - residual_debt["curve"]
        }



class AssetQualityRatingScoresEstimator:

    @staticmethod
    def get_asset_quality_rating_scores(asset_data, telemetry_data, number_of_units):
        operational_use_curve = AssetQualityRatingScoresEstimator._get_operational_use_curve(asset_data, telemetry_data)
        quality_rating_curve = AssetQualityRatingScoresEstimator._get_quality_rating_curve(asset_data, telemetry_data, operational_use_curve)
        return {
            "quality_rating_curve" : quality_rating_curve,
            "operational_use_curve" : operational_use_curve
        }

    @staticmethod
    def _get_operational_use_curve(asset_data, telemetry_data):
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        baseline = asset_data["category_data"]["useful_life_hours"] / len(telemetry_data["mean_curve"])
        for i in range(len(telemetry_data["mean_curve"])):
            lower_bound_curve.append(int((telemetry_data["lower_bound_curve"][i]/baseline) * 100))
            upper_bound_curve.append(int((telemetry_data["upper_bound_curve"][i]/baseline) * 100))
            mean_curve.append(int((telemetry_data["mean_curve"][i]/baseline) * 100))

        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }

    @staticmethod
    def _get_quality_rating_curve(asset_data, telemetry_data, operational_use_curve):
        ''' Quality is the normalized [-5,5] ratio between 1/average last 30 days of operational use'''
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        i_min = 0
        i_max = 0
        interval = 30
        min_value = -5
        max_value = 5
        for i in range(len(telemetry_data["mean_curve"])):
            lower_bound_curve.append(((1/np.mean(operational_use_curve["lower_bound_curve"][i_min : i_max + 1])) * (max_value - min_value)) + min_value)
            upper_bound_curve.append(((1/np.mean(operational_use_curve["upper_bound_curve"][i_min : i_max + 1])) * (max_value - min_value)) + min_value)
            mean_curve.append(((1/np.mean(operational_use_curve["mean_curve"][i_min : i_max + 1])) * (max_value - min_value)) + min_value)
            i_max += 1
            if i >= interval - 1:
                i_min += 1

        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }



class EsgRatingScoresEstimator:

    @staticmethod
    def get_esg_ratings_scores(asset_data, telemetry_data, number_of_units):
        energy_consumed = EsgRatingScoresEstimator._get_energy_consumed(asset_data, telemetry_data)
        return {
            "footprint_curve" : EsgRatingScoresEstimator._get_footprint_curve(asset_data, telemetry_data, energy_consumed),
            "energy_consumed" : energy_consumed,
            "environmental_risk_indicators" : EsgRatingScoresEstimator._get_environmental_risk_indicators(asset_data)
        }

    @staticmethod
    def _get_energy_consumed(asset_data, telemetry_data):
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        power = asset_data["category_data"]["power_kw"]
        for i in range(len(telemetry_data["mean_curve"])):
            lower_bound_curve.append(telemetry_data["lower_bound_curve"][i] * power)
            upper_bound_curve.append(telemetry_data["upper_bound_curve"][i] * power)
            mean_curve.append(telemetry_data["mean_curve"][i] * power)

        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }

    @staticmethod
    def _get_footprint_curve(asset_data, telemetry_data, energy_consumed):
        ''' WRONG Consider Constant Emission Factor TODO'''
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        emission_factor = asset_data["city_data"]["carbon_intensity_gCO2eq_kWh"]
        for i in range(len(telemetry_data["mean_curve"])):
            lower_bound_curve.append(emission_factor * energy_consumed["lower_bound_curve"][i])
            upper_bound_curve.append(emission_factor * energy_consumed["upper_bound_curve"][i])
            mean_curve.append(emission_factor * energy_consumed["mean_curve"][i])


        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }

    @staticmethod
    def _get_environmental_risk_indicators(asset_data):
        pass


class StrategyAdvisorScoresEstimator:

    @staticmethod
    def get_strategy_advisors_scores(leasing_risk_scores, asset_quality_rating_scores, esg_ratings_scores):
        pass

class AssetScoresEstimator:

    @staticmethod
    def get_scores(asset_data, telemetry_data, number_of_units):
        leasing_risk = LeasingRiskScoresEstimator.get_leasing_risk_scores(asset_data, telemetry_data, number_of_units)

        asset_quality_rating = AssetQualityRatingScoresEstimator.get_asset_quality_rating_scores(asset_data, telemetry_data, number_of_units)

        esg_rating = EsgRatingScoresEstimator.get_esg_ratings_scores(asset_data, telemetry_data, number_of_units)

        strategy_advisor = StrategyAdvisorScoresEstimator.get_strategy_advisors_scores(leasing_risk, asset_quality_rating, esg_rating)

        return {
            "number_of_units": number_of_units,
            "asset_quality_rating": asset_quality_rating,
            "leasing_risk": leasing_risk,
            "esg_rating": esg_rating,
            "strategy_advisor": strategy_advisor
        }





