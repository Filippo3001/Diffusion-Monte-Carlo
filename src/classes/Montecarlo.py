import numpy as np
import random as rd

#class or function???

class MonteCarlo:
    def __init__(self, f, h):
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