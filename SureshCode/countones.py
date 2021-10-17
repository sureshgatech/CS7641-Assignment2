import os
import sys
import csv
from time import clock
import time

sys.path.append(r"E:\CS7641\Assignment2\ABAGAIL.jar")

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
from array import array


def train(alg_func, alg_name, ef, iters):
    ef.resetFunctionEvaluationCount()
    fit = FixedIterationTrainer(alg_func,iters)
    FILE_NAME = alg_name + "_countones.csv"
    OUTPUT_FILE = os.path.join("E:\CS7641\Assignment2\SureshCode\data", FILE_NAME)
    with open(OUTPUT_FILE, "wb") as results:
        writer = csv.writer(results, delimiter=',')
        writer.writerow(["iters", "fevals", "fitness","time"])
        times = [0]
        for i in range(0,iters, 1):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            # print str(i) + ", " + str(ef.getFunctionEvaluations()) + ", " + str(ef.value(alg_func.getOptimal()))
            writer.writerow([i, ef.getFunctionEvaluations() - i, ef.value(alg_func.getOptimal()),times[-1]])

    print(alg_name + ": " + str(ef.value(alg_func.getOptimal())))
    print("Function Evaluations: " + str(ef.getFunctionEvaluations() - iters))
    print("Iters: " + str(iters))
    print("####")


"""
Commandline parameter(s):
   none
"""

N=800
fill = [2] * N
ranges = array('i', fill)

ef = CountOnesEvaluationFunction()
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

"""
rhc = RandomizedHillClimbing(hcp)
train(rhc, "RHC", ef, 300)

#started with 10000 iterations and ended with 100
sa = SimulatedAnnealing(1E11, .95, hcp)
train(sa, "SA_1", ef, 100)
sa = SimulatedAnnealing(1E11, .9, hcp)
train(sa, "SA_2", ef, 100)
sa = SimulatedAnnealing(1E11, .75, hcp)
train(sa, "SA_3", ef, 100)
sa = SimulatedAnnealing(1E11, .6, hcp)
train(sa, "SA_4", ef, 100)
sa = SimulatedAnnealing(1E11, .3, hcp)
train(sa, "SA_5", ef, 100)
sa = SimulatedAnnealing(1E11, .1, hcp)
train(sa, "SA_6", ef, 100)
"""

#tried 100 first
ga = StandardGeneticAlgorithm(800, 400, 100, gap)
train(ga, "GA_1", ef, 3000)
ga = StandardGeneticAlgorithm(800, 200, 100, gap)
train(ga, "GA_2", ef, 3000)
ga = StandardGeneticAlgorithm(800, 200, 50, gap)
train(ga, "GA_3", ef, 3000)
ga = StandardGeneticAlgorithm(400, 100, 50, gap)
train(ga, "GA_4", ef, 3000)
ga = StandardGeneticAlgorithm(400, 200, 100, gap)
train(ga, "GA_5", ef, 3000)
ga = StandardGeneticAlgorithm(400, 100, 50, gap)
train(ga, "GA_6", ef, 3000)

"""
mimic = MIMIC(800, 400, pop)
train(mimic,"MIMIC", ef, 100)
print ("MIMIC_1: " + str(ef.value(mimic.getOptimal())))

mimic = MIMIC(800, 200, pop)
train(mimic,"MIMIC_2", ef, 100)
print ("MIMIC_2: " + str(ef.value(mimic.getOptimal())))

mimic = MIMIC(800, 100, pop)
train(mimic,"MIMIC_3", ef, 100)
print ("MIMIC: " + str(ef.value(mimic.getOptimal())))

mimic = MIMIC(400, 200, pop)
train(mimic,"MIMIC_4", ef, 100)
print ("MIMIC: " + str(ef.value(mimic.getOptimal())))

mimic = MIMIC(200, 100, pop)
train(mimic,"MIMIC_5", ef, 100)
print ("MIMIC: " + str(ef.value(mimic.getOptimal())))

mimic = MIMIC(200, 50, pop)
train(mimic,"MIMIC_6", ef, 100)
print ("MIMIC: " + str(ef.value(mimic.getOptimal())))
"""


#mimic = MIMIC(50, 10, pop)
#fit = FixedIterationTrainer(mimic, 100)
#train()
#print ("MIMIC: " + str(ef.value(mimic.getOptimal())))
