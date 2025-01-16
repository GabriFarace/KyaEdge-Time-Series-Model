import numpy as np
from prophet import Prophet
import math

from sintethic_data_generation.utils import create_prophet_dataframe


class LeasingRiskScoresEstimator:

    @staticmethod
    def get_leasing_risk_scores(asset_data, telemetry_data, number_of_units):
        '''Return the scores associated to the Leasing Risk task'''
        remarketing_value_curve = LeasingRiskScoresEstimator._get_remarketing_value_curve(asset_data, telemetry_data)
        residual_debt = LeasingRiskScoresEstimator._get_residual_debt(asset_data, telemetry_data)
        return {
            "remarketing_value_curve" : remarketing_value_curve,
            "residual_debt" : residual_debt,
            "gap_curve" : LeasingRiskScoresEstimator._get_gap_curve(remarketing_value_curve, residual_debt)
        }


    @staticmethod
    def _get_residual_debt(asset_data, telemetry_data):
        ''' Return the residual debt : account also for the difference cost - contract amount'''
        residual_debt = []
        current_debt = asset_data["contract_data"]["contract_amount"] - asset_data["contract_data"]["contract_upfront_payment"] + (asset_data["purchase_cost"] - asset_data["contract_data"]["contract_amount"])
        daily_subtraction = current_debt / len(telemetry_data["mean_curve"])
        for i in range(len(telemetry_data["mean_curve"])):
            residual_debt.append(round(current_debt,2))
            current_debt = current_debt - daily_subtraction

        return {"curve" : residual_debt}

    @staticmethod
    def _get_remarketing_value_curve(asset_data, telemetry_data):
        ''' Return the market value using the cost approach'''

        #TODO no obsolescence here
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
            lower_bound_curve.append(round(((cost - residual_value) - (cost - residual_value) * (applied_life_lower/useful_life)) + residual_value, 2))
            upper_bound_curve.append(round(((cost - residual_value) - (cost - residual_value) * (applied_life_upper/useful_life)) + residual_value, 2))
            mean_curve.append(round(((cost - residual_value) - (cost - residual_value) * (applied_life_mean/useful_life)) + residual_value, 2))
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
        ''' Return the gap curve as difference between the remarketing value and the residual debt'''
        return {
            "lower_bound_curve" : (np.array(remarketing_value_curve["lower_bound_curve"]) - np.array(residual_debt["curve"])).tolist(),
            "mean_curve" : (np.array(remarketing_value_curve["mean_curve"]) - np.array(residual_debt["curve"])).tolist(),
            "upper_bound_curve" : (np.array(remarketing_value_curve["upper_bound_curve"]) - np.array(residual_debt["curve"])).tolist()
        }


class AssetQualityRatingScoresEstimator:

    @staticmethod
    def get_asset_quality_rating_scores(asset_data, telemetry_data, number_of_units):
        ''' Return the  scores associated to the quality rating task'''
        operational_use_curve = AssetQualityRatingScoresEstimator._get_operational_use_curve(asset_data, telemetry_data)
        quality_rating_curve = AssetQualityRatingScoresEstimator._get_quality_rating_curve(asset_data, telemetry_data, operational_use_curve)
        return {
            "quality_rating_curve" : quality_rating_curve,
            "operational_use_curve" : operational_use_curve
        }

    @staticmethod
    def _get_operational_use_curve(asset_data, telemetry_data):
        ''' Return the operational use curve'''

        # TODO baseline is fixed and set to useful_life/number of days
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        num_units = 365 * asset_data["category_data"]["useful_life_years"]
        baseline = asset_data["category_data"]["useful_life_hours"] / num_units
        for i in range(len(telemetry_data["mean_curve"])):
            if telemetry_data["lower_bound_curve"][i] > 0:
                lower_bound_curve.append(int((telemetry_data["lower_bound_curve"][i]/baseline) * 100))
            else:
                lower_bound_curve.append(1)

            if telemetry_data["upper_bound_curve"][i] > 0:
                 upper_bound_curve.append(int((telemetry_data["upper_bound_curve"][i]/baseline) * 100))
            else:
                upper_bound_curve.append(1)

            if telemetry_data["mean_curve"][i] > 0:
                mean_curve.append(int((telemetry_data["mean_curve"][i]/baseline) * 100))
            else:
                mean_curve.append(1)

        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }

    @staticmethod
    def _get_quality_rating_curve(asset_data, telemetry_data, operational_use_curve):
        ''' Return the quality rating curve. Quality is the normalized [-5,5] ratio between (1/average history of operational use)'''
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        i_max = 0
        baseline = asset_data["category_data"]["useful_life_hours"] / len(telemetry_data["mean_curve"])
        max_ou = math.ceil((24 / baseline) * 100)
        h = 1 / max_ou

        def custom_function(x):
            ''' piecewise function that return output values between -5 and 5'''
            if 0.01 <= x <= 1:
                return round(5 * ( (x - 0.01) / 0.99), 2)
            elif 0 <= x < 0.01:
                return round(-5 + ( (5 / ( 0.01 - h) ) * (x - h)), 2)
            else:
                print(x)
                raise ValueError(f"Input must be in the range [{h}, 1].")

        for i in range(len(telemetry_data["mean_curve"])):
            lower_bound_curve.append(custom_function(1/np.mean(operational_use_curve["lower_bound_curve"][0 : i_max + 1])))
            upper_bound_curve.append(custom_function(1/np.mean(operational_use_curve["upper_bound_curve"][0 : i_max + 1])))
            mean_curve.append(custom_function(1/np.mean(operational_use_curve["mean_curve"][0 : i_max + 1])))
            i_max += 1


        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }


class EsgRatingScoresEstimator:

    @staticmethod
    def get_esg_ratings_scores(asset_data, telemetry_data, number_of_units):
        ''' Return the scores associated to the ESG rating task'''
        energy_consumed = EsgRatingScoresEstimator._get_energy_consumed(asset_data, telemetry_data)
        return {
            "footprint_curve" : EsgRatingScoresEstimator._get_footprint_curve(asset_data, telemetry_data, energy_consumed),
            "energy_consumed" : energy_consumed,
            "environmental_risk_indicators" : EsgRatingScoresEstimator._get_environmental_risk_indicators(asset_data)
        }

    @staticmethod
    def _get_energy_consumed(asset_data, telemetry_data):
        ''' Return the energy consumed curve by multiplying the telemetry bu the power'''
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        power = asset_data["category_data"]["power_kw"]
        for i in range(len(telemetry_data["mean_curve"])):
            lower_bound_curve.append(round(telemetry_data["lower_bound_curve"][i] * power, 2))
            upper_bound_curve.append(round(telemetry_data["upper_bound_curve"][i] * power, 2))
            mean_curve.append(round(telemetry_data["mean_curve"][i] * power, 2))

        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }

    @staticmethod
    def _get_footprint_curve(asset_data, telemetry_data, energy_consumed):
        ''' Return the footprint curve'''
        # todo WRONG Consider Constant Emission Factor
        lower_bound_curve = []
        upper_bound_curve = []
        mean_curve = []
        emission_factor = asset_data["city_data"]["carbon_intensity_gCO2eq_kWh"]/1000
        for i in range(len(telemetry_data["mean_curve"])):
            lower_bound_curve.append(round(emission_factor * energy_consumed["lower_bound_curve"][i], 2))
            upper_bound_curve.append(round(emission_factor * energy_consumed["upper_bound_curve"][i], 2))
            mean_curve.append(round(emission_factor * energy_consumed["mean_curve"][i], 2))


        return {
            "lower_bound_curve" : lower_bound_curve,
            "mean_curve" : mean_curve,
            "upper_bound_curve" : upper_bound_curve
        }

    @staticmethod
    def _get_environmental_risk_indicators(asset_data):
        ''' Return the environmental risk indicators for the asset'''
        protective_measures = asset_data["esg_inputs"]["protective_measures"]

        def get_final_risk(risk, protective_measures_number):
            ''' Lower the risk by one level for each 3 protective measures'''
            risk_lowered_by = protective_measures_number / 3
            # LOW 1 or less than 0
            # MEDIUM 2
            # HIGH 3
            final_risk = risk - risk_lowered_by

            # If no protective measures, increase risk level
            if protective_measures_number == 0:
                final_risk = final_risk + 1

            if final_risk <= 0:
                return 0
            elif final_risk >= 5:
                return 5
            else:
                return final_risk

        flood_hazard = get_final_risk(asset_data["city_data"]["flood_hazard"], len(protective_measures["flood_hazard_protective_measures"]))
        landslide_hazard = get_final_risk(asset_data["city_data"]["landslide_hazard"], len(protective_measures["landslide_hazard_protective_measures"]))
        climatic_hazard = get_final_risk(asset_data["city_data"]["climatic_hazard"], len(protective_measures["climatic_hazard_protective_measures"]))
        seismic_hazard = get_final_risk(asset_data["city_data"]["seismic_hazard"], len(protective_measures["seismic_hazard_protective_measures"]))

        summary_rating = int((flood_hazard + landslide_hazard + climatic_hazard + seismic_hazard)/4)

        return {
            "flood_hazard" : flood_hazard,
            "landslide_hazard" : landslide_hazard,
            "climatic_hazard" : climatic_hazard,
            "seismic_hazard" : seismic_hazard,
            "summary_rating" : summary_rating
        }


class StrategyAdvisorScoresEstimator:

    @staticmethod
    def get_strategy_advisors_scores(leasing_risk_scores, asset_quality_rating_scores, esg_ratings_scores, number_of_units):
        ''' Return the scores associated to the Strategy Advisors task'''
        asset_quality_rating_task = StrategyAdvisorScoresEstimator._get_quality_rating_advice(asset_quality_rating_scores, number_of_units)
        leasing_risk_task = StrategyAdvisorScoresEstimator._get_leasing_risk_advice(leasing_risk_scores, number_of_units)
        esg_rating_task = StrategyAdvisorScoresEstimator._get_esg_rating_advice(esg_ratings_scores, number_of_units)

        count = 0
        if "CRITICO" in asset_quality_rating_task["status"]:
            count += 1
        if "CRITICO" in leasing_risk_task["status"]:
            count += 1
        if "CRITICO" in esg_rating_task["status"]:
            count += 1

        if count == 0:
            priority = 0
        elif count == 1:
            priority = 1
        elif count == 2:
            priority = 3
        else:
            priority = 5

        return {
            "priority_value" : priority,
            "asset_quality_rating_task" : asset_quality_rating_task,
            "leasing_risk_task" : leasing_risk_task,
            "esg_rating_task" : esg_rating_task
        }

    @staticmethod
    def _get_quality_rating_advice(asset_quality_rating_scores, number_of_units):
        ''' Return the quality rating task advice. Check the quality now and at the end of the contract'''
        low_quality_now = asset_quality_rating_scores["quality_rating_curve"]["mean_curve"][number_of_units - 1] < 0
        low_quality_end = asset_quality_rating_scores["quality_rating_curve"]["mean_curve"][-1] < 0

        if low_quality_now and low_quality_end:
            status = "CRITICO : Sulla base del livello di utilizzo rilevato e del modello previsionale di stima, l’asset è significativamente sovrautilizzato e si sta rapidamente deteriorando"
            advice = "Si suggerisce una proposta di sostituzione/ rinnovo tecnologico per ottimizzare profitto prima che il gap remarketing/debito cliente scenda sotto lo zero"
        elif low_quality_now:
            status = "MEDIO : L'asset è sovrautilizzato al momento ma sulla base del modello previsionale non lo sarà al termine del contratto"
            advice = "Nessuna azione suggerita"
        elif low_quality_end:
            status = "MEDIO : L'asset non è sovrautilizzato al momento ma sulla base del modello previsionale lo sarà al termine del contratto"
            advice = "Si suggerisce una proposta di sostituzione/ rinnovo tecnologico per ottimizzare profitto prima che il gap remarketing/debito cliente scenda sotto lo zero"
        else:
            status = "Il valore della qualità dell'asset è ritenuto ACCETTABILE"
            advice = "Nessuna azione suggerita"

        return {
            "status" : status,
            "advice" : advice
        }

    @staticmethod
    def _get_leasing_risk_advice(leasing_risk_scores, number_of_units):
        ''' Return the leasing risk advice. Check the GAP now and at the end of the contract'''
        low_quality_now = leasing_risk_scores["gap_curve"]["mean_curve"][number_of_units - 1] < 0
        low_quality_end = leasing_risk_scores["gap_curve"]["mean_curve"][-1] < 0

        if low_quality_now and low_quality_end:
            status = "CRITICO : Sulla base del valore di mercato corrente e del modello previsionale di stima, il GAP rilevato è minore di 0"
            advice = "Si suggerisce una misura che vada a portare il GAP ad un valore maggiore di 0"
        elif low_quality_now:
            status = "MEDIO : L'asset ha un GAP minore di 0 al momento ma sulla base del modello previsionale non lo sarà al termine del contratto"
            advice = "Nessuna azione suggerita"
        elif low_quality_end:
            status = "MEDIO : L'asset ha un GAP maggiore di 0 al momento ma sulla base del modello previsionale sarà minore di 0 al termine del contratto"
            advice = "Si suggerisce una misura che vada a portare il GAP ad un valore maggiore di 0 al termine del contratto"
        else:
            status = "Il valore del rischio di locazione è ritenuto ACCETTABILE"
            advice = "Nessuna azione suggerita"

        return {
            "status": status,
            "advice": advice
        }

    @staticmethod
    def _get_esg_rating_advice(esg_ratings_scores, number_of_units):
        ''' Return the esg rating advice. Check the environmental risk indicators  (if some hazard has high risk or not) now and at the end of the contract'''

        indicators = [
            esg_ratings_scores["environmental_risk_indicators"]["flood_hazard"],
            esg_ratings_scores["environmental_risk_indicators"]["landslide_hazard"],
            esg_ratings_scores["environmental_risk_indicators"]["climatic_hazard"],
            esg_ratings_scores["environmental_risk_indicators"]["seismic_hazard"]
        ]
        high_risk = any(indicator == "high" for indicator in indicators)

        if high_risk:
            status = "CRITICO : Sulla base degli indicatori esg, è presente un alto rischio ambientale"
            advice = "Si suggerisce una misura protettiva appropriata per l'asset che riduca il rischio ambientale"
        else:
            status = "Il valore degli indicatori esg è ritenuto ACCETTABILE"
            advice = "Nessuna azione suggerita"

        return {
            "status": status,
            "advice": advice
        }


class AssetScoresEstimator:

    @staticmethod
    def get_scores(asset_data, telemetry_data, number_of_units):
        ''' Return the scores of the 4 task (Leasing Risk, Asset Quality Rating, ESG Rating and Strategy Advisor)'''
        leasing_risk = LeasingRiskScoresEstimator.get_leasing_risk_scores(asset_data, telemetry_data, number_of_units)

        asset_quality_rating = AssetQualityRatingScoresEstimator.get_asset_quality_rating_scores(asset_data, telemetry_data, number_of_units)

        esg_rating = EsgRatingScoresEstimator.get_esg_ratings_scores(asset_data, telemetry_data, number_of_units)

        strategy_advisor = StrategyAdvisorScoresEstimator.get_strategy_advisors_scores(leasing_risk, asset_quality_rating, esg_rating, number_of_units)

        return {
            "number_of_units": number_of_units,
            "asset_quality_rating": asset_quality_rating,
            "leasing_risk": leasing_risk,
            "esg_rating": esg_rating,
            "strategy_advisor": strategy_advisor
        }

    @staticmethod
    def get_asset_expected_usage(asset_data, telemetry_history):
        hours_on_duty = round(float(np.mean(telemetry_history) * 365 ), 2)
        return {
            "hours_on_duty" : hours_on_duty,
            "kwh" : int(hours_on_duty * asset_data["category_data"]["power_kw"])
        }

    @staticmethod
    def get_standard_average_asset_expected_usage(asset_data):
        num_units = 365 * asset_data["category_data"]["useful_life_years"]
        baseline = asset_data["category_data"]["useful_life_hours"] / num_units
        hours_on_duty = int(baseline) * 365
        return {
            "hours_on_duty" : hours_on_duty,
            "kwh" : int(hours_on_duty * asset_data["category_data"]["power_kw"])
        }


class AggregateScoresEstimator:

    @staticmethod
    def aggregate_scores_lessor(assets_list, lessor_id):
        ''' Return the aggregates scores for the given lessor'''
        scores = [asset["scores"] for asset in assets_list]

        return {
            "lessor_id": lessor_id,
            "num_asset" : len(assets_list), #TODO adjust (only today asset and not expired)
            "asset_quality_rating" : AggregateScoresEstimator._get_aggregates_asset_quality_rating(scores),
            "esg_rating" : AggregateScoresEstimator._get_aggregates_esg_rating(scores),
            "leasing_risk" : AggregateScoresEstimator._get_aggregates_leasing_risk(scores),
            "strategy_advisor" : AggregateScoresEstimator._get_aggregates_strategy_advisor(scores)
        }

    @staticmethod
    def _get_aggregates_asset_quality_rating(scores):
        ''' Return the aggregates asset quality rating scores. HIGH if less than -2, MEDIUM if between -2 and 0, LOW if > 0'''

        now_quality_scores = [score["asset_quality_rating"]["quality_rating_curve"]["mean_curve"][score["number_of_units"] - 1] for score in scores]
        high_risk_now = sum( value < -2 for value in now_quality_scores)
        medium_risk_now = sum((0 > value > -2) for value in now_quality_scores)
        low_risk_now = sum( value > 0 for value in now_quality_scores)
        six_months_quality_scores = []
        for score in scores:
            if len(score["asset_quality_rating"]["quality_rating_curve"]["mean_curve"]) > score["number_of_units"] + 6:
                six_months_quality_scores.append(score["asset_quality_rating"]["quality_rating_curve"]["mean_curve"][score["number_of_units"] + 5])

        high_risk_six_months = sum( value < -2 for value in six_months_quality_scores)
        medium_risk_six_months = sum((0 > value > -2) for value in six_months_quality_scores)
        low_risk_six_months = sum( value > 0 for value in six_months_quality_scores)

        return {
            "high_risk_now": high_risk_now,
            "medium_risk_now": medium_risk_now,
            "low_risk_now": low_risk_now,
            "high_risk_six_months": high_risk_six_months,
            "medium_risk_six_months": medium_risk_six_months,
            "low_risk_six_months": low_risk_six_months
        }

    @staticmethod
    def _get_aggregates_esg_rating(scores):
        ''' Return the aggregates esg rating scores'''

        # First extract the environmental risk indicators
        number_flood = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        number_landslide = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        number_climatic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        number_seismic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for score in scores:
            indicators = score["esg_rating"]["environmental_risk_indicators"]
            number_flood[indicators["flood_hazard"]] += 1
            number_landslide[indicators["landslide_hazard"]] += 1
            number_climatic[indicators["climatic_hazard"]] += 1
            number_seismic[indicators["seismic_hazard"]] += 1



        # Now process footprint curves
        footprint_curves = []
        energy_consumed_curves = []

        max_length_before_today = 0
        max_length_after_today = 0
        for score in scores:
            length_before_today = score["number_of_units"]
            length_after_today = len(score["esg_rating"]["footprint_curve"]["mean_curve"]) - length_before_today
            if length_before_today > max_length_before_today:
                max_length_before_today = length_before_today
            if length_after_today > max_length_after_today:
                max_length_after_today = length_after_today

        # Fill curves with zeros before and after today so that they all have the same length and range of dates
        for score in scores:
            length_before_today = score["number_of_units"]
            footprint_curve = score["esg_rating"]["footprint_curve"]["mean_curve"]
            energy_consumed = score["esg_rating"]["energy_consumed"]["mean_curve"]
            length_after_today = len(footprint_curve) - length_before_today

            adjusted_footprint_curve = \
                ([ 0. for i in range(max_length_before_today - length_before_today)] +
                 footprint_curve +
                 [ 0. for i in range(max_length_after_today - length_after_today)])

            adjusted_energy_curve = \
                ([ 0. for i in range(max_length_before_today - length_before_today)] +
                 energy_consumed +
                 [ 0. for i in range(max_length_after_today - length_after_today)])

            footprint_curves.append(adjusted_footprint_curve)
            energy_consumed_curves.append(adjusted_energy_curve)


        footprint_curves = np.array(footprint_curves)
        energy_consumed_curves = np.array(energy_consumed_curves)

        sum_footprint_curves = footprint_curves.sum(axis=0)
        sum_energy_curves = energy_consumed_curves.sum(axis=0)
        footprint_curve_co2_mwh = []

        for i in range(sum_footprint_curves.size):
            if sum_footprint_curves[i] > 0:
                footprint_curve_co2_mwh.append(sum_footprint_curves[i] / sum_energy_curves[i] * 1000)
            else:
                footprint_curve_co2_mwh.append(0)

        footprint_curve_co2_euro_day = []
        # todo footprint_curve_co2_euro_day

        return {
            "number_of_units" : max_length_before_today,
            "footprint_curve_co2_mwh" : footprint_curve_co2_mwh,
            "footprint_curve_co2_euro_day" : footprint_curve_co2_euro_day,
            "environmental_risk_indicators" : {
                "flood_hazard" :{
                    0 : number_flood[0],
                    1 : number_flood[1],
                    2 : number_flood[2],
                    3 : number_flood[3],
                    4 : number_flood[4],
                    5 : number_flood[5]
                },
                "landslide_hazard" :{
                    0 : number_landslide[0],
                    1 : number_landslide[1],
                    2 : number_landslide[2],
                    3 : number_landslide[3],
                    4 : number_landslide[4],
                    5 : number_landslide[5]
                },
                "climatic" :{
                    0 : number_climatic[0],
                    1 : number_climatic[1],
                    2 : number_climatic[2],
                    3 : number_climatic[3],
                    4 : number_climatic[4],
                    5 : number_climatic[5]
                },
                "seismic" :{
                    0 : number_seismic[0],
                    1 : number_seismic[1],
                    2 : number_seismic[2],
                    3 : number_seismic[3],
                    4 : number_seismic[4],
                    5 : number_seismic[5]
                }
            }
        }


    @staticmethod
    def _get_aggregates_leasing_risk(scores):
        ''' Return the aggregates leasing risk scores'''
        gap_assets_today = []
        gap_assets_six_months = []

        for score in scores:
            number_of_units = score["number_of_units"]
            gap_curve = score["leasing_risk"]["gap_curve"]["mean_curve"]
            gap_assets_today.append(gap_curve[number_of_units - 1])
            if number_of_units + 6 <= len(gap_curve):
                gap_assets_six_months.append(gap_curve[number_of_units - 1 + 6])

        # Define the bin edges
        # todo define other possible bins
        bins = [-20_000, -10_000, -5_000, -2_500, 0, 2_500, 5_000, 10_000, 20_000]

        # Count the number of elements in each bin
        hist_today, _ = np.histogram(gap_assets_today, bins=bins)
        hist_six_months, _ = np.histogram(gap_assets_six_months, bins=bins)

        # Add counts for values outside the range
        below_range_today = sum(v < bins[0] for v in gap_assets_today)
        above_range_today = sum(v > bins[-1] for v in gap_assets_today)

        below_range_six_months = sum(v < bins[0] for v in gap_assets_six_months)
        above_range_six_months = sum(v > bins[-1] for v in gap_assets_six_months)

        # Combine counts into a single list
        empirical_distribution_today = [below_range_today] + hist_today.tolist() + [above_range_today]
        empirical_distribution_six_months = [below_range_six_months] + hist_six_months.tolist() + [above_range_six_months]

        return {
            "global_GAP_today" : sum(gap_assets_today),
            "global_GAP_six_months" : sum(gap_assets_six_months),
            "gap_distribution_today" : empirical_distribution_today,
            "gap_distribution_six_months" : empirical_distribution_six_months
        }

    @staticmethod
    def _get_aggregates_strategy_advisor(scores):
        ''' Return the aggregates strategy advisor scores'''
        priority_values = [score["strategy_advisor"]["priority_value"] for score in scores]

        number_high_priority = sum(v == 5 for v in priority_values)
        number_low_priority = sum(v == 3 for v in priority_values)
        number_assets_usage_to_be_investigated = sum(v == 1 for v in priority_values)
        number_current_situation_under_control = sum(v == 0 for v in priority_values)

        return {
            "number_high_priority" : number_high_priority,
            "number_low_priority" : number_low_priority,
            "number_assets_usage_to_be_investigated" : number_assets_usage_to_be_investigated,
            "number_current_situation_under_control" : number_current_situation_under_control
        }



def get_forecasted_telemetry(telemetry_data, future_periods, sum_maximum, today, start_date=None):
    ''' Use prophet to forecast the future telemetry'''
    df = create_prophet_dataframe(telemetry_data, start_date)
    m = Prophet(n_changepoints=200)
    m.add_seasonality(name='monthly', period=30, fourier_order=5)
    m.add_seasonality(name='bimonthly', period=60, fourier_order=7)
    m.fit(df)
    future = m.make_future_dataframe(periods=future_periods)
    forecast = m.predict(future)
    #m.plot(forecast)


    # Filter rows where 'ds' > today
    filtered_df = forecast[forecast['ds'] > today]

    # Extract the columns as lists
    yhat_list = np.clip(filtered_df['yhat'], 0, 24).tolist()
    yhat_lower_list = np.clip(filtered_df['yhat_lower'], 0, 24).tolist()
    yhat_upper_list = np.clip(filtered_df['yhat_upper'], 0, 24).tolist()



    def build_sum(telemetry, sum_value):
        '''' Constraint the sum of the time series to be sum_value by truncating to 0 the series from the time step where the sum is reached'''

        telemetry_copy = np.array(telemetry)
        # Step 1: Compute the cumulative sum
        cumulative_sum = np.cumsum(telemetry_copy)

        # Step 2: Find the index where cumulative sum exceeds the limit
        truncation_index = np.argmax(cumulative_sum > sum_value) if np.any(cumulative_sum > sum_value) else -1

        # Step 3: Truncate the series
        if truncation_index != -1:  # Only truncate if the limit is surpassed
            telemetry_copy[truncation_index:] = 0

        return telemetry_copy.tolist()

    return  {
        "lower_bound_curve": build_sum(telemetry_data + yhat_lower_list, sum_maximum),
        "upper_bound_curve": build_sum(telemetry_data + yhat_upper_list, sum_maximum),
        "mean_curve": build_sum(telemetry_data + yhat_list, sum_maximum),
    }




