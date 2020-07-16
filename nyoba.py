import json


"""with open('preprocessing.json') as json_file:
    data = json.load(json_file)
    for i in data['token']:
        print(i)
    json_file.close()"""

file = open('preprocessing.json',)

data = json.load(file)

for i in data:
    print(i['token'])

file.close()