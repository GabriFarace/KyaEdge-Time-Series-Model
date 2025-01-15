import json

from sintethic_data_generation.estimators import AggregateScoresEstimator


def aggregates_scores_main():
    with open("data_months.json", "r") as f:
        data = json.load(f)

    lessors_assets_scores_list = {}

    for asset_data in data:

        lessor_id = asset_data["lessor_id"]
        if lessor_id not in lessors_assets_scores_list.keys():
            lessors_assets_scores_list[lessor_id] = []

        lessors_assets_scores_list[lessor_id].append(asset_data)

    for lessor_id in lessors_assets_scores_list.keys():
        print(len(lessors_assets_scores_list[lessor_id]))
        aggregates_lessor = AggregateScoresEstimator.aggregate_scores_lessor(lessors_assets_scores_list[lessor_id], lessor_id)

        with open(f'lessor{lessor_id}_data/aggregates.json', 'w') as json_file:
            json.dump(aggregates_lessor, json_file, indent=4)

        with open(f'lessor{lessor_id}_data/data.json', 'w') as json_file:
            json.dump(lessors_assets_scores_list[lessor_id], json_file, indent=4)








if __name__ == '__main__':
    aggregates_scores_main()