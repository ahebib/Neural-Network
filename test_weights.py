#Ammar Hebib
#Test Weights

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

def Open_and_Load_Synaptic_Weights(numSensors,numMotors):
    #opens and loads best weights for testing
    weightsFileName = 'C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/best_weights.dat'
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


def fitness3_get(synapses):
    weightsFileName = 'C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/weights.dat'
    fitFileName = 'C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/fits.dat'
    time.sleep(0.5)
    if(os.path.isfile(fitFileName)):
        os.remove(fitFileName)
    time.sleep(0.2)
    Send_Synapse_Weights_ToFile(synapses,weightsFileName)
    #starts the robot
    os.startfile('C:/Users/Ammar/Documents/University of Vermont/Spring 2013/CS 206 Evolutionary Robotics/Final Project/bullet-2.81-rev2613/bullet-2.81-rev2613/App_RagdollDemo_vs2010_debug.exe')
    while(os.path.isfile(fitFileName) == False):
        time.sleep(0.6)
    
    with open(fitFileName, 'r') as w:
        fitness = w.readline()
    
    if(os.path.isfile(fitFileName)):
        os.remove(fitFileName)

    if(os.path.isfile(weightsFileName)):
        os.remove(weightsFileName)

    
    return fitness


def main():
        numSensors = 4
        numMotors = 8
        numGenerations = 0
        parent = Open_and_Load_Synaptic_Weights(numSensors,numMotors)
        parentFit = fitness3_get(parent)
        for currentGeneration in range(numGenerations):
            parentFit = fitness3_get(parent)
            childFit = parentFit
            print currentGeneration, parentFit, childFit    
main()
