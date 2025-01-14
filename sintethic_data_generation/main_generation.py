import json

with open('categories.json', 'w') as json_file:
    json.dump(categories, json_file, indent=4)