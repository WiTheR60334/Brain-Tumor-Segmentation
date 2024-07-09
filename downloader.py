#this file will download the dataset from synpase.org
import synapseclient 

syn = synapseclient.Synapse() 
#paste your authToken below in order to downlaod the BraTS dataset
syn.login(authToken="") 

# Obtain a pointer and download the data 
syn60084146 = syn.get(entity='syn60084146') 

# Get the path to the local copy of the data file 
filepath = syn60084146.path

print(f'Data file downloaded to: {filepath}')
