# Literature Case Study
# MILP Script
# Author: Andreas Juhl Sørensen
# 2024

#Import PYOMO
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

#Import packages for data and visualisation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################################################
##DICTIONARIES are created to carry data in a transparent and general manner
#States (as K for material)
K = {
        'FA':   {'Cap': 500, 'Ini': 500, 'Price':  0},
        'FB':   {'Cap': 500, 'Ini': 500, 'Price':  0},
        'FC':   {'Cap': 500, 'Ini': 500, 'Price':  0},
        'HotA': {'Cap': 100, 'Ini':   0, 'Price': -100},
        'AB':   {'Cap': 200, 'Ini':   0, 'Price': -100},
        'BC':   {'Cap': 150, 'Ini':   0, 'Price': -100},
        'E':    {'Cap': 100, 'Ini':   0, 'Price': -100},
        'P1':   {'Cap': 500, 'Ini':   0, 'Price': 10},
        'P2':   {'Cap': 500, 'Ini':   0, 'Price': 10},
    }

#State-to-Task nodes with feed amount/stoichiometry (as KtI for material to task)
KtI = {
        ('FA',   'Heating'):    {'Feed': 1.0},
        ('FB',   'R1'):         {'Feed': 0.5},
        ('FC',   'R1'):         {'Feed': 0.5},
        ('FC',   'R3'):         {'Feed': 0.2},
        ('HotA', 'R2'):         {'Feed': 0.4},
        ('AB',   'R3'):         {'Feed': 0.8},
        ('BC',   'R2'):         {'Feed': 0.6},
        ('E',    'Separation'): {'Feed': 1.0},
    }

#Task-to-State nodes with task processing time and conversion coefficient
ItK = {
        ('Heating', 'HotA'):  {'tau': 1, 'rho': 1.0},
        ('R2', 'P1'):         {'tau': 2, 'rho': 0.4},
        ('R2', 'AB'):         {'tau': 2, 'rho': 0.6},
        ('R1', 'BC'):         {'tau': 2, 'rho': 1.0},
        ('R3', 'E') :         {'tau': 1, 'rho': 1.0},
        ('Separation', 'AB'): {'tau': 2, 'rho': 0.1},
        ('Separation', 'P2'): {'tau': 1, 'rho': 0.9},
    }

#Units able to perform specific tasks node (as JI_union) with capacity
#and gamma of performing task
JI_union = {
        ('Heater', 'Heating'):    {'Betamin': 0, 'Betamax': 100, 'gamma': 1},
        ('Reactor1', 'R1'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 1},
        ('Reactor1', 'R2'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 1},
        ('Reactor1', 'R3'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 1},
        ('Reactor2', 'R1'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 1},
        ('Reactor2', 'R2'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 1},
        ('Reactor2', 'R3'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 1},
        ('Still', 'Separation'):  {'Betamin': 0, 'Betamax': 200, 'gamma': 1},
    }
###############################################################################

##TASKS

#All tasks in a set
I = set([i for (j,i) in JI_union])

#Materials that are produced by task i
Kplus = {i: set() for i in I}
for (i,k) in ItK:
    Kplus[i].add(k)

#Materials that are consumed by task i
Kminus = {i: set() for i in I}
for (k,i) in KtI:
    Kminus[i].add(k)

#Input fraction (feed) of task i from material k
Feed = {(i,k): KtI[(k,i)]['Feed'] for (k,i) in KtI}

#Output fraction (conversion coefficient) of task i to material k
rho = {(i,k): ItK[(i,k)]['rho'] for (i,k) in ItK}

#Time to release of material k from task i (tauK), and task processing time (tau)
tauK = {(i,k): ItK[(i,k)]['tau'] for (i,k) in ItK}
tau = {i: max([tauK[(i,k)] for k in Kplus[i]]) for i in I}

#Units capable of task i
Ji = {i: set() for i in I}
for (j,i) in JI_union:
    Ji[i].add(j)

###############################################################################

##STATES

#Tasks producing material k
Iplus = {k: set() for k in K}
for (i,k) in ItK:
    Iplus[k].add(i)

#Tasks consuming material k
Iminus = {k: set() for k in K}
for (k,i) in KtI:
    Iminus[k].add(i)

#Storage capacity/maximum inventory of material k
Sk_max = {k: K[k]['Cap'] for k in K}

###############################################################################

##UNITS

#All units in a set
J = set([j for (j,i) in JI_union])

#Tasks able to be carried out in unit j
Ij = {j: set() for j in J}
for (j,i) in JI_union:
    Ij[j].add(i)
    
#Minimum and maximum capacity
Betamin = {(i,j):JI_union[(j,i)]['Betamin'] for (j,i) in JI_union}
Betamax = {(i,j):JI_union[(j,i)]['Betamax'] for (j,i) in JI_union}

###############################################################################

##MODELLING: VARIABLES AND OBJECTIVE

#Create model environment
model = pyo.ConcreteModel()

#Planning horizon (H), time interval (tgap) and time (T)
H = 10
tgap = 1
T = tgap*np.array(range(0,int(1/tgap)*H+1))

#Decision variable Wijt
model.W = pyo.Var(I,J,T,domain=pyo.Boolean)

#Batch size decision cariable Bijt
model.B = pyo.Var(I,J,T,domain=pyo.NonNegativeReals)

#Inventory variable
model.S = pyo.Var(K.keys(),T, domain=pyo.NonNegativeReals)

#Value of inventory
model.SVal = pyo.Var(domain=pyo.NonNegativeReals)
model.SValcon = pyo.Constraint(expr = model.SVal == sum([K[k]['Price']*model.S[k,H] for k in K]))

#Cost of operation
model.OpCost = pyo.Var(domain=pyo.NonNegativeReals)
model.OpCostcon = pyo.Constraint(expr = model.OpCost == 
                                 sum([JI_union[(j,i)]['gamma']*model.W[i,j,t] 
                                      for i in I for j in Ji[i] for t in T])) 

#Objective function defined as maximisation of inventory value minus cost of operation
model.obj = pyo.Objective(expr = model.SVal - model.OpCost, sense = pyo.maximize)

###############################################################################

##MODELLING: EQUATIONS AND CONSTRAINTS

#Create constraint environment
model.con = pyo.ConstraintList()

#Equation 3: Inventory development over time (material mass balance)
for k in K.keys():
    eq = K[k]['Ini']
    for t in T:
        #Production
        for i in Iplus[k]:
            for j in Ji[i]:
                if t >= tauK[(i,k)]:
                    eq = eq + rho[(i,k)]*model.B[i,j,max(T[T <= t-tauK[(i,k)]])]
        #Consumption
        for i in Iminus[k]:
            eq = eq - rho[(i,k)]*sum([model.B[i,j,t] for j in Ji[i]])
        model.con.add(model.S[k,t] == eq)
        eq = model.S[k,t] 

#Constraint 1: A unit can only take one task at a time
for j in J:
    for t in T:
        eq = 0
        for i in Ij[j]:
            for n in T:
                if n >= (t-tau[i]+1) and n <= t:
                    eq = eq + model.W[i,j,n]
        model.con.add(eq <= 1)            

#Constraint 2: Unit capacity constraint
for t in T:
    for j in J:
        for i in Ij[j]:
            model.con.add(model.W[i,j,t]*Betamin[i,j] <= model.B[i,j,t])
            model.con.add(model.B[i,j,t] <= model.W[i,j,t]*Betamax[i,j]) 

#Constraint 3: Material/inventory capacity constraint
model.Scon = pyo.Constraint(K.keys(), T, rule = lambda model, k, t: model.S[k,t] <= Sk_max[k])

###############################################################################

##SOLVE MODEL AND VISUALISE

#Solve the model with PYOMO optimisation
SolverFactory('cplex').solve(model).write()

#Visualise solution in Gantt chart
plt.figure(figsize=(12,6))

gap = H/500
idx = 1
lbls = []
ticks = []
for j in sorted(J):
    idx = idx - 1
    for i in sorted(Ij[j]):
        idx = idx - 1
        ticks.append(idx)
        lbls.append("{0:s} -> {1:s}".format(j,i))
        plt.plot([0,H],[idx,idx],lw=20,alpha=.3,color='y')
        for t in T:
            if model.W[i,j,t]() > 0:
                plt.plot([t+gap,t+tau[i]-gap], [idx,idx],'b', lw=20, solid_capstyle='butt')
                txt = "{0:.2f}".format(model.B[i,j,t]())
                plt.text(t+tau[i]/2, idx, txt, color='white', weight='bold', ha='center', va='center')
plt.xlim(0,H)
plt.gca().set_yticks(ticks)
plt.gca().set_yticklabels(lbls);