# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
import csv
import os
import sys
from time import clock
import time

sys.path.append(r"E:\CS7641\Assignment2\ABAGAIL.jar")

import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import util.ABAGAILArrays as ABAGAILArrays

from array import array
import shared.ConvergenceTrainer as ConvergenceTrainer

def train(alg_func, alg_name, ef, iters):
    ef.resetFunctionEvaluationCount()
    fit = FixedIterationTrainer(alg_func,iters)
    FILE_NAME = alg_name + "_tsm.csv"
    OUTPUT_FILE = os.path.join("E:\CS7641\Assignment2\SureshCode\data", FILE_NAME)
    with open(OUTPUT_FILE, "wb") as results:
        writer = csv.writer(results, delimiter=',')
        writer.writerow(["iters", "fevals", "fitness", "time"])
        times = [0]
        for i in range(0, iters,1):
            start = clock()
            fit.train()
            #print("Function Evaluations: " + str(ef.getFunctionEvaluations() - iters))
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            # print str(i) + ", " + str(ef.getFunctionEvaluations()) + ", " + str(ef.value(alg_func.getOptimal()))
            writer.writerow([i, ef.getFunctionEvaluations() - i, ef.value(alg_func.getOptimal()), times[-1]])

    print(alg_name + ": " + str(ef.value(alg_func.getOptimal())))
    print("Function Evaluations: " + str(ef.getFunctionEvaluations() - iters))
    print("Iters: " + str(iters))
    print("####")


"""
Commandline parameter(s):
    none
"""

# set N value.  This is the number of points
N = 100
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

"""
rhc = RandomizedHillClimbing(hcp)
train(rhc, "RHC", ef, 2000)
print ("RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal())))
print ("Route:")
path = []
for x in range(0,N):
    path.append(rhc.getOptimal().getDiscrete(x))
print (path)


sa = SimulatedAnnealing(1E12, .5, hcp)
train(sa, "SA", ef, 300)
print ("SA Inverse of Distance: " + str(ef.value(sa.getOptimal())))
print ("Route:")
path = []
for x in range(0,N):
    path.append(sa.getOptimal().getDiscrete(x))
print (path)


ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
train(ga, "GA", ef, 1000)
print ("GA Inverse of Distance: " + str(ef.value(ga.getOptimal())))
print ("Route:")
path = []
for x in range(0,N):
    path.append(ga.getOptimal().getDiscrete(x))
print (path)


# for mimic we use a sort encoding
ef = TravelingSalesmanSortEvaluationFunction(points);
fill = [N] * N
ranges = array('i', fill)
odd = DiscreteUniformDistribution(ranges);
df = DiscreteDependencyTree(.1, ranges); 
pop = GenericProbabilisticOptimizationProblem(ef, odd, df);
"""

"""
rhc = RandomizedHillClimbing(hcp)
train(rhc, "RHC", ef, 3000)
print ("RHC_1: " + str(ef.value(rhc.getOptimal())))
#started with 10000 iterations and ended with 100

sa = SimulatedAnnealing(1E11, .95, hcp)
train(sa, "SA_1", ef, 3000)
print ("sa_1: " + str(ef.value(sa.getOptimal())))

sa = SimulatedAnnealing(1E11, .9, hcp)
train(sa, "SA_2", ef, 3000)
print ("sa_2: " + str(ef.value(sa.getOptimal())))

sa = SimulatedAnnealing(1E11, .75, hcp)
train(sa, "SA_3", ef, 3000)
print ("sa_3: " + str(ef.value(sa.getOptimal())))

sa = SimulatedAnnealing(1E11, .6, hcp)
train(sa, "SA_4", ef, 3000)
print ("sa_4: " + str(ef.value(sa.getOptimal())))
sa = SimulatedAnnealing(1E11, .3, hcp)
train(sa, "SA_5", ef, 3000)
print ("sa_5: " + str(ef.value(sa.getOptimal())))
sa = SimulatedAnnealing(1E11, .1, hcp)
train(sa, "SA_6", ef, 3000)
print ("sa_6: " + str(ef.value(sa.getOptimal())))

#tried 100 first

ga = StandardGeneticAlgorithm(100, 50, 25, gap)
train(ga, "GA_1", ef, 3000)
print ("ga_1: " + str(ef.value(ga.getOptimal())))
ga = StandardGeneticAlgorithm(100, 40, 20, gap)
train(ga, "GA_2", ef, 3000)
print ("ga_2: " + str(ef.value(ga.getOptimal())))
ga = StandardGeneticAlgorithm(100, 20, 10, gap)
train(ga, "GA_3", ef, 3000)
print ("ga_3: " + str(ef.value(ga.getOptimal())))
ga = StandardGeneticAlgorithm(50, 50, 25, gap)
train(ga, "GA_4", ef, 3000)
print ("ga_4: " + str(ef.value(ga.getOptimal())))
ga = StandardGeneticAlgorithm(50, 25, 10, gap)
train(ga, "GA_5", ef, 3000)
print ("ga_5: " + str(ef.value(ga.getOptimal())))
ga = StandardGeneticAlgorithm(50, 20, 5, gap)
train(ga, "GA_6", ef, 3000)
print ("ga_6: " + str(ef.value(ga.getOptimal())))


mimic = MIMIC(100, 50, pop)
train(mimic,"MIMIC", ef, 200)
print ("MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal())))
print ("Route:")
path = []
optimal = mimic.getOptimal()
fill = [0] * optimal.size()
ddata = array('d', fill)
for i in range(0,len(ddata)):
    ddata[i] = optimal.getContinuous(i)
order = ABAGAILArrays.indices(optimal.size())
ABAGAILArrays.quicksort(ddata, order)
print (order)
"""

print ("MIMIC_ started: " )
fill = [N] * N
ranges = array('i', fill)
ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscreteUniformDistribution(ranges)
df = DiscreteDependencyTree(0.1, ranges);
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
mimic = MIMIC(100, 100, pop)
train(mimic,"MIMIC", ef, 200)
print ("MIMIC_1: " + str(ef.value(mimic.getOptimal())))

df = DiscreteDependencyTree(0.5, ranges);
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
mimic = MIMIC(100, 50, pop)
train(mimic,"MIMIC_2", ef, 200)
print ("MIMIC_2: " + str(ef.value(mimic.getOptimal())))

df = DiscreteDependencyTree(.8, ranges);
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
mimic = MIMIC(100, 25, pop)
train(mimic,"MIMIC_3", ef, 200)
print ("MIMIC: " + str(ef.value(mimic.getOptimal())))

df = DiscreteDependencyTree(0.1, ranges);
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
mimic = MIMIC(50, 30, pop)
train(mimic,"MIMIC_4", ef, 200)
print ("MIMIC: " + str(ef.value(mimic.getOptimal())))

df = DiscreteDependencyTree(0.5, ranges);
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
mimic = MIMIC(50, 20, pop)
train(mimic,"MIMIC_5", ef, 200)
print ("MIMIC: " + str(ef.value(mimic.getOptimal())))

df = DiscreteDependencyTree(0.8, ranges);
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
mimic = MIMIC(25, 10, pop)
train(mimic,"MIMIC_6", ef, 200)
print ("MIMIC: " + str(ef.value(mimic.getOptimal())))
