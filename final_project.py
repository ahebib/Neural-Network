#Ammar Hebib
#CS 206
#Final Project.py

from numpy import*
from pylab import*
import matplotlib.pyplot as plt
from scipy import*
from random import*
import random
from copy import deepcopy
import os
import time
import sys

def CreateMatrix(rows,columns):
    array = zeros( (rows,columns) )
    return array

def MatrixRandom(Array_Name):
    #inputs random values into matrix between 0-1
    random.seed()
    array_rows = len(Array_Name)
    array_columns = (size(Array_Name[0]))
    for row in range(array_rows):
        for column in range(array_columns):
            Array_Name[row,column] = random.uniform(-1,1)
    return Array_Name

##def Update(valueMatrix, synMatrix ):
##    for i in range( 1,valueMatrix.shape[0] ):
##        valueMatrix[i] = np.dot( synMatrix, valueMatrix[i-1] )
##        for j in range( 0, valueMatrix.shape[1] ):
##            if ( valueMatrix[i][j] > 1 ):
##                valueMatrix[i][j] = 1
##            elif ( valueMatrix[i][j] < 0 ):
##                valueMatrix[i][j] = 0
##    return valueMatrix

def Update(neuronValues,synapses,i):
    numcol = len(neuronValues[0])
    for j in range(numcol):
        temp = 0
        for k in range(0,9):
            temp += neuronValues[i-1,k]*synapses[j,k]

        if(temp > 1):
            temp = 1
        elif(temp < 0):
            temp = 0
        neuronValues[i][j] = temp
    return neuronValues

def FitnessCalc(parent):
    neuronValues = zeros(shape=(10,10))
    for i in range(10):
        neuronValues[0][i] = 0.5

    for i in range(1,10):
        neuronValues = Update(neuronValues,parent,i)

    actualNeuronValues = neuronValues[9,:]
    
    desiredNeuronValues = VectorCreate(10)
    for j in range(0,10,2):
        desiredNeuronValues[j]=1
        
    d = MeanDistance(actualNeuronValues,desiredNeuronValues)
    fit = 1 - d
   
    return fit,neuronValues
    
def FitnessCalc2(array):
    #calclates fitness of the array
    temper = np.zeros((10,10))
    for n in range(10):
        temper[0,n] = 0.5
        
    for i in range(1,10):
        neuronValues = Update(temper,array,i)

    diff=0.0
    for i in range(0,9):
        for j in range(0,9):
            diff= diff + abs(neuronValues[i,j]-neuronValues[i,j+1])
            diff= diff + abs(neuronValues[i+1,j]-neuronValues[i,j])

    diff=diff/(2*9*9)

    return diff,neuronValues


def fitness3_get(synapses):
    weightsFileName = 'C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/weights.dat'
    fitFileName = 'C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/fits.dat'
    if(os.path.isfile(fitFileName)):
        os.remove(fitFileName)
    Send_Synapse_Weights_ToFile(synapses,weightsFileName)
    #starts the robot
    os.startfile('C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/bullet-2.81-rev2613/bullet-2.81-rev2613/App_RagdollDemo_vs2010_debug.exe')
    while(os.path.isfile(fitFileName) == False):
        time.sleep(1.5)
    
    with open(fitFileName, 'r') as w:
        fitness = w.readline()
    
    if(os.path.isfile(fitFileName)):
        os.remove(fitFileName)

    if(os.path.isfile(weightsFileName)):
        os.remove(weightsFileName)

    
    return fitness

def Open_and_Load_Start_Synaptic_Weights(numSensors,numMotors):
    #opens and loads start weights
    weightsFileName = 'C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/start_weights.dat'
    synapse = zeros( (numSensors,numMotors) )
    with open(weightsFileName, 'r') as w:
        for m in range(len(synapse)):
            for n in range(len(synapse[0])):
                synapse[m][n] = w.readline()
    return synapse


def Open_and_Load_Synaptic_Weights(numSensors,numMotors):
    #opens and loads best weights for testing
    weightsFileName = 'C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/best_weights_log.dat'
    synapse = zeros( (numSensors,numMotors) )
    with open(weightsFileName, 'r') as w:
        for m in range(len(synapse)):
            for n in range(len(synapse[0])):
                synapse[m][n] = w.readline()
    return synapse


def Send_Synapse_Weights_ToFile(synapses,weightsFileName):
    #writes weights 
    with open('weights.dat', 'w') as f:
        for i in range(len(synapses)): #go through rows
            for j in range(len(synapses[0])): #go through columns
                f.write('%f\n' % synapses[i][j])

def Send_Fitness_to_File(fitness):
    #records fitness vector
    with open('FitnessMatrix.dat', 'w') as f:
        for i in range(len(fitness)): #go through row
                f.write('%f\n' % fitness[i])

def Send_Weights_to_Log_File(synapses):
    #this records the best synaptic weights for the run only
    with open('best_weights_log.dat', 'w') as f:
        for i in range(len(synapses)): #go through rows
            for j in range(len(synapses[0])): #go through columns
                f.write('%f\n' % synapses[i][j])

def Send_Fitness_Generation_Weight_to_File(currentGeneration, fitness, weights):
    #records current generation, fitness vector, and weights
    with open('cumulative_results.dat', 'a+') as f:
        f.write('----- Current Generation: ' + str(currentGeneration) + ' -----\n\n')

    with open('cumulative_results.dat', 'a+') as f:
            f.write('fitness: ' + fitness + '\n\n')

    with open('cumulative_results.dat', 'a+') as f:
        for i in range(len(weights)): #go through rows
            for j in range(len(weights[0])): #go through columns
                f.write('%f\n' % weights[i][j])
        f.write('\n')

    
def MatrixPerturb(array, p):
    #make deepcopy of array
    #make random numn
    #if random num < .05
    #check your elements in array if > .05 and if lower new random

    arrayRows = len(array)
    arrayColumns = (len(array[0]))

    arrayCopy = deepcopy(array)
        
    for i in range(arrayRows):
        for r in range(arrayColumns):
            if(np.random.random() < p):
                arrayCopy[i][r] = random.uniform(-1,1)
    return arrayCopy

def VectorCreate(numNeurons):
    matrix_list = zeros((numNeurons),dtype='f')
    return matrix_list

def MeanDistance(v1,v2):
    dist = 0
    for i in range(len(v1)):
        dist += sum(pow(v1[i]-v2[i],2))
    dist = sqrt(dist)/sqrt(10)
    
    return dist

def plotFigureA(neuronValues):
    plt.imshow(neuronValues, cmap=plt.get_cmap('gray'), aspect= 'auto',interpolation= 'nearest')
    plt.show()
        
def main():
        numSensors = 4
        numMotors = 8
        numGenerations = 200
        parent = Open_and_Load_Start_Synaptic_Weights(numSensors,numMotors)
        #parent = CreateMatrix(numSensors,numMotors)
        #parent = MatrixRandom(parent)
        #parentfitness = parentfit now
        parentFit = fitness3_get(parent)
##        parentFitness,neuronValues = FitnessCalc2(parent)
##        #plotFigureA(neuronValues)
        Fitness = CreateMatrix(1,numGenerations)
        Fitness = Fitness[0,:]
        Send_Fitness_to_File(Fitness)
        for currentGeneration in range(numGenerations):
            #print currentGeneration, parentFitness
            Fitness[currentGeneration] = parentFit
            child = MatrixPerturb(parent,0.05)
            childFit = fitness3_get(child)
            print currentGeneration, parentFit, childFit
            Send_Fitness_Generation_Weight_to_File(currentGeneration, childFit, child)
##            childFitness,neuronValues = FitnessCalc2(child)
            if(childFit > parentFit):
                parent = deepcopy(child)
                parentFit = childFit
                Send_Weights_to_Log_File(parent)
            Fitness[currentGeneration] = parentFit
            Send_Fitness_to_File(Fitness)
        
##        plotFigureA(neuronValues)
##        plt.plot(Fitness)
##        plt.show()
        
main()
    
