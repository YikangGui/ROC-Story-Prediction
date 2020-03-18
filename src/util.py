import csv
import json

# csvFilePath = '../SupportingMaterials/ROC-Story-Cloze-Data.csv'
csvFilePath = '../SupportingMaterials/ROC-Story-Cloze-Val.csv'

contextJsonFilePath = '../SupportingMaterials/context.json'
endingJsonFilePath = '../SupportingMaterials/ending.json'
trainFilePath = '../SupportingMaterials/test.txt'
context = {}
ending = {}

with open(csvFilePath, 'r', encoding='utf-8') as csvFile:
    csvReader = csv.DictReader(csvFile)
    for i, rows in enumerate(csvReader):
        # context[i] = rows['InputSentence1'] + ' ' + rows['InputSentence2'] + ' ' + rows['InputSentence3'] + ' ' + rows[
        #     'InputSentence4'] + ' ' + rows['RandomFifthSentenceQuiz1'] + ' ' + rows['RandomFifthSentenceQuiz2']
        # ending[i] = [rows['RandomFifthSentenceQuiz1'], rows['RandomFifthSentenceQuiz2']]
        with open(trainFilePath, 'a') as trainFile:
            tmp = {}
            tmp['context'] = rows['InputSentence1']
            trainFile.write(json.dumps(tmp)+'\n')
            tmp['context'] = rows['InputSentence2']
            trainFile.write(json.dumps(tmp)+'\n')
            tmp['context'] = rows['InputSentence3']
            trainFile.write(json.dumps(tmp)+'\n')
            tmp['context'] = rows['InputSentence4']
            trainFile.write(json.dumps(tmp)+'\n')
            tmp['context'] = rows['RandomFifthSentenceQuiz1']
            trainFile.write(json.dumps(tmp)+'\n')
            tmp['context'] = rows['RandomFifthSentenceQuiz2']
            trainFile.write(json.dumps(tmp)+'\n')

# with open(contextJsonFilePath, 'w') as jsonFile:
#     if jsonFile.writable():
#         jsonFile.write(json.dumps(context, indent=4))
#     else:
#         print('Cannot write context json file!')
#
# with open(endingJsonFilePath, 'w') as jsonFile:
#     if jsonFile.writable():
#         jsonFile.write(json.dumps(ending, indent=4))
#     else:
#         print('Cannot write ending json file!')

if __name__ == '__main__':
    print(context)
