import json

with open ('./bayes.json') as f:
    bayes = json.load(f)

print(bayes)

print('Bayes classroom is located on the %dth floor in the %s classroom' % (bayes['Floor'], bayes['Location']))

type(bayes['Floor'])


if bayes['isActive']:
    print('Bayes is Active')

print('The number of students in the class is %d \nThe number of instructors is %d' % (len(bayes['Students']),len(bayes['Instructors'])))


most_languages = []

for i in range(0,len(bayes['Instructors'])):
    most_languages.append(len(bayes['Instructors'][i]['favoriteLanguages']))

for i in most_languages:
    if i == max(most_languages):
        for n in range(0,len(bayes['Instructors'])):
            if len(bayes['Instructors'][n]['favoriteLanguages']) == i:
                print(bayes['Instructors'][n]['name'])


len(bayes['Instructors'][2]['favoriteLanguages'])
len(bayes['Instructors'][2]['favoriteLanguages']) 


