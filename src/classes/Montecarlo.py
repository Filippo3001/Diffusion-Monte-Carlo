import numpy as np
import random as rd
    
# Remake the class with a different apporach.
# The method will be applied only to the variables.
# it works with with f(*ndarray), function of the form f(*par, *ndarray) need to be wrapped with partial method

class MonteCarlo:
    accepted_points = []

    def  __init__(self, f, startingPoint):
        self.f = f
        self.point = startingPoint    # current point from which new points wil be generated

    #  The generate method will add nMoves new valid generated points to the list of accepted_points
    def generate(self, nMoves, nThermMoves, metroStep):
        rd.seed()   # Initialize random generator
        rng = np.random.default_rng()
        f_r  = self.f(self.point)

        for i in range(nMoves):              #Loop of moves
            for j in range(nThermMoves):     #Thermalization loop
                #trialStep = np.zeros_like(self.point)
                #for k in range(len(self.point)):
                #    trialStep[k] = self.point[k] + (rd.random() - 0.5) * metroStep  #modify so that it works with both number variables or lists.
                trialStep = self.point + (rng.random(self.point.shape) - 0.5) * metroStep
                new_f_r = self.f(trialStep)
                if new_f_r > f_r:            #always accept when the new probability is higher 
                    self.point = trialStep
                    f_r = new_f_r
                else:
                    if rd.random() < new_f_r/f_r:   #if the new probability is lower than accept with probability new_probability/old_probability
                        self.point = trialStep
                        f_r = new_f_r
            
            self.accepted_points.append(self.point)
        
    def clear(self):
        self.accepted_points.clear()

    def setStartingPoint(self, startingPoint):
        self.point = startingPoint
    
    # The evaluate work with functions h(ndarray), as in f(*par, ndarray).
    def evaluate(self, *args):
        points_to_evaluate = np.array(self.accepted_points).transpose() # convert the list of points to a (m,...) numpy array
        SE = np.zeros_like(args, dtype = float)
        SE2 = np.zeros_like(args, dtype = float)
        for i in range(len(SE)):
            SE[i] = np.sum(args[i](points_to_evaluate))
            SE2[i] = np.sum(args[i](points_to_evaluate)**2)
        
        N  = len(self.accepted_points)
        mean = SE / N
        sigma = np.sqrt(((SE2/N) - (SE/N)**2) / N)

        return mean, sigma


