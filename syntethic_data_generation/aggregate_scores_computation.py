import json
import os

from syntethic_data_generation.estimators import AggregateScoresEstimator



def aggregates_scores_main():
    ''' Read the file data_months and compute the aggregation using the estimators.AggregateScoresEstimator'''
    with open("data_months.json", "r") as f:
        data = json.load(f)

    os.makedirs("json_data", exist_ok=True)
    lessors_assets_scores_list = {}

    for asset_data in data:

        lessor_id = asset_data["lessor_id"]
        if lessor_id not in lessors_assets_scores_list.keys():
            lessors_assets_scores_list[lessor_id] = []

        lessors_assets_scores_list[lessor_id].append(asset_data)

    for lessor_id in lessors_assets_scores_list.keys():
        print(len(lessors_assets_scores_list[lessor_id]))
        aggregates_lessor = AggregateScoresEstimator.aggregate_scores_lessor(lessors_assets_scores_list[lessor_id], lessor_id)

        os.makedirs(f"json_data/lessor{lessor_id}_data/", exist_ok=True)
        with open(f'json_data/lessor{lessor_id}_data/aggregates_scores_{lessor_id}.json', 'w') as json_file:
            json.dump(aggregates_lessor, json_file, indent=4)

        with open(f'json_data/lessor{lessor_id}_data/data_{lessor_id}.json', 'w') as json_file:
            json.dump(lessors_assets_scores_list[lessor_id], json_file, indent=4)








if __name__ == '__main__':
    aggregates_scores_main()