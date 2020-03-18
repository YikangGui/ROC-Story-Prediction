import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("predictedFile", help ="a file with predicted classifications in each line of the file")
parser.add_argument("goldFile", help ="a file with gold classifications in each line of the file")
parser.add_argument("-v", "--verbose", required = False,help="this will print out all incorrect comparisons made to help check if there are formatting issues", action = "store_true")
args = parser.parse_args()

predictedResults = []
with open(args.predictedFile, 'r') as f:
	for line in f:
		val = line.strip()
		if(val != ""):
			predictedResults.append( val )
		
goldResults = []
with open(args.goldFile, 'r') as f:
	for line in f:
		val = line.strip()
		if(val != ""):
			goldResults.append( val )
		

#do some quick checks to make sure the input is valid
if(len(predictedResults) != len(goldResults)):
	print("Error: number of elements between inputs does not match please check that your input is formatted properly")
	raise SystemExit

if(args.verbose):
	print("printing out the incorrect matches")
#computes the accuracy of the two lists
correctCount = 0
for i in range(len(goldResults)):
	result = goldResults[i]==predictedResults[i]
	correctCount += int(result)
	if(not result and args.verbose):
		print("Predicted Value (%s) ----- Gold Value (%s)" %(predictedResults[i], goldResults[i]))
	
	
print("Accuracy: %.5f" %( float(correctCount)/float(len(goldResults))))
