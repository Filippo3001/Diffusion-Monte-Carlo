import numpy as np
import random as rd

#class or function???

#first version of the class
class MonteCarlov1:
    def __init__(self, f):
        self.f = f  # importance sampling (maybe rename variables)

    def integrate(self, h, r, nMoves, nThermMoves, metroStep):
        rd.seed()   # Initialize random generator
        f_r  = self.f(r)
        SE = 0
        SE2 = 0

        for i in range(nMoves):              #Loop of moves
            for j in range(nThermMoves):     #Thermalization loop
                trialStep = r + (rd.random() - 0.5) * metroStep
                new_f_r = self.f(trialStep)
                if new_f_r > f_r:            #always accept when the new probability is higher 
                    r = trialStep
                    f_r = new_f_r
                else:
                    if rd.random() < new_f_r/f_r:   #if the new probability is lower than accept with probability new_probability/old_probability
                        r = trialStep
                        f_r = new_f_r
                
            new_h = h(r)   #compute the local value for  h
            SE += new_h         #update  accumulation of local values
            SE2 += new_h**2
        
        average = SE/nMoves
        sigma = np.sqrt(SE2/nMoves - (SE/nMoves)**2)/nMoves

        return average, sigma
    
    def generate(self, r, nMoves, nThermMoves, metroStep):
        rd.seed()   # Initialize random generator
        f_r  = self.f(r)
        accepted_r = []

        for i in range(nMoves):              #Loop of moves
            for j in range(nThermMoves):     #Thermalization loop
                trialStep = r + (rd.random() - 0.5) * metroStep
                new_f_r = self.f(trialStep)
                if new_f_r > f_r:            #always accept when the new probability is higher 
                    r = trialStep
                    f_r = new_f_r
                else:
                    if rd.random() < new_f_r/f_r:   #if the new probability is lower than accept with probability new_probability/old_probability
                        r = trialStep
                        f_r = new_f_r
            
            accepted_r.append(r)
        
        return accepted_r
    

# Remake the class with a different apporach.
# The importance sampling will be of the form f(par, x) where both par and x are arrays of parameters and variables respectively.
# The method will be applied only to the variables.
# Will bee changed to work with f(*vectors), function of the form f(par, x) need to be wrapped with partial method

class MonteCarlov2:
    accepted_points = []

    def  __init__(self, f, r):
        self.f = f
        self.r = r

    #  The generate method will add nMoves new valid generated points to the list of accepted_points
    def generate(self, nMoves, nThermMoves, metroStep):
        rd.seed()   # Initialize random generator
        f_r  = self.f(self.r)

        for i in range(nMoves):              #Loop of moves
            for j in range(nThermMoves):     #Thermalization loop
                trialStep = np.zeros[len(self.r)]
                for k in len(self.r):
                    trialStep[k] = self.r[k] + (rd.random() - 0.5) * metroStep[k]
                new_f_r = self.f(trialStep)
                if new_f_r > f_r:            #always accept when the new probability is higher 
                    self.r = trialStep
                    f_r = new_f_r
                else:
                    if rd.random() < new_f_r/f_r:   #if the new probability is lower than accept with probability new_probability/old_probability
                        self.r = trialStep
                        f_r = new_f_r
            
            self.accepted_points.append(self.r)
        
    def clear(self):
        self.accepted_points.clear()

    def setStartingPoint(self, r):
        self.r = r
    
    # The evaluate work with functions h(x) where x is an array with equal size to the one in f(par, x).
    def evaluate(self, *args):
        SE = []
        SE2 = []
        for i in range(args):
            for p in self.accepted_points:
                SE[i] += args[i](p)
                SE2[i] += args[i](p)**2
        
        N  = len(self.accepted_points)
        sigma = np.sqrt((SE2/N) - (SE/N)**2)

