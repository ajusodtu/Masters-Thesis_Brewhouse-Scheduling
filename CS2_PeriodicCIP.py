# Brewery Case Study - Periodic CIP Implementation
# MILP Script
# Author: Andreas Juhl Sørensen
# 2024

# %%

#Import PYOMO
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

#Import packages for data and visualisation
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
##DICTIONARIES are created to carry data in a transparent and general manner
#States (as K for material)
K = {
        'Malt':   {'Cap': 100000,'Ini': 100000,'nu':  0},
        'Water':  {'Cap': 400000,'Ini': 400000,'nu':  0},
        'HM':     {'Cap': 520,  'Ini':   0,  'nu':  -0.01},
        'SG':     {'Cap': 100000,'Ini':   0,  'nu':  0},
        'W':      {'Cap': 510,  'Ini':   0,  'nu':  -0.01},
        'BW':     {'Cap': 440,  'Ini':   0,  'nu':  -0.01},
        'CW':     {'Cap': 100000,'Ini':   0,  'nu':  0.03},
        'Waste':  {'Cap': 100000,'Ini':   0,  'nu':  0},
        'CIPin':  {'Cap': 10000,'Ini': 10000,'nu':  0},
        'CIPout': {'Cap': 10000,'Ini':   0,  'nu':  0},
    }

#State-to-Task nodes with feed amount/stoichiometry (as KtI for material to task)
KtI = {
        ('Malt',  'MillMashing'):   {'xi': 0.25},
        ('Water', 'MillMashing'):   {'xi': 0.75},
        ('HM',    'Lautering'):     {'xi': 0.75},
        ('Water', 'Lautering'):     {'xi': 0.25},
        ('W',     'Boiling'):       {'xi': 1.0},
        ('BW',    'WhirlCooling'):  {'xi': 1.0},
        ('CIPin', 'aCIP'):           {'xi': 1.0},
    }

#Task-to-State nodes with task processing time and conversion coefficient
ItK = {
        ('MillMashing', 'HM'):      {'tau': 135,  'rho': 0.9025},
        ('MillMashing', 'Waste'):   {'tau': 135,  'rho': 0.0975},
        ('Lautering', 'W'):         {'tau': 90,   'rho': 0.7315},
        ('Lautering', 'SG'):        {'tau': 90,   'rho': 0.23},
        ('Lautering', 'Waste'):     {'tau': 90,   'rho': 0.0385},
        ('Boiling', 'BW') :         {'tau': 105,  'rho': 0.857375},
        ('Boiling', 'Waste') :      {'tau': 105,  'rho': 0.142625},
        ('WhirlCooling', 'CW'):     {'tau': 60,   'rho': 0.9025},
        ('WhirlCooling', 'Waste'):  {'tau': 60,   'rho': 0.0975},
        ('aCIP', 'CIPout'):          {'tau': 90,   'rho': 1},
    }

#Units able to perform specific tasks node (as JI_union) with capacity
#and gamma of performing task
JI_union = {
        ('MillMash 1', 'MillMashing'):      {'Betamin': 0, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('MillMash 2', 'MillMashing'):      {'Betamin': 0, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 1', 'Lautering'):      {'Betamin': 0, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 2', 'Lautering'):      {'Betamin': 0, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Wort Kettle', 'Boiling'):         {'Betamin': 0, 'Betamax': 450, 'gamma': 0.01, 'etamax':3,'theta':1},
        ('WhirlCool', 'WhirlCooling'):      {'Betamin': 0, 'Betamax': 411, 'gamma': 0.01, 'etamax':3,'theta':1},
        
        #CIP
        ('Wort Kettle', 'aCIP'):             {'Betamin': 1, 'Betamax': 1,  'gamma': 0.01, 'etamax':3,'theta':0},
        ('WhirlCool', 'aCIP'):               {'Betamin': 1, 'Betamax': 1,  'gamma': 0.01, 'etamax':3,'theta':0},
    }

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

#Input fraction (xi) of task i from material k
xi = {(i,k): KtI[(k,i)]['xi'] for (k,i) in KtI}

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
H = 18*60
tgap = 15
T = tgap*np.array(range(0,int(1/tgap*H)+1))

#Decision variable Wijt
model.W = pyo.Var(I,J,T,domain=pyo.Boolean)

#Batch size decision cariable Bijt
model.B = pyo.Var(I,J,T,domain=pyo.NonNegativeReals)

#Lifted variable handling: M is inventory of unit j at time t
model.M = pyo.Var(J, T, domain=pyo.NonNegativeReals)

#Inventory variable
model.S = pyo.Var(K.keys(),T, domain=pyo.NonNegativeReals)

#Cost of operation
model.OpCost = pyo.Var(domain=pyo.NonNegativeReals)
model.OpCostcon = pyo.Constraint(expr = model.OpCost == 
                                 sum([JI_union[(j,i)]['gamma']*model.W[i,j,t] 
                                      for i in I for j in Ji[i] for t in T]))

#Value of inventory
model.SVal = pyo.Var(domain=pyo.NonNegativeReals)
model.SValcon = pyo.Constraint(expr = model.SVal == sum([K[k]['nu']*model.S[k,t] for k in K for t in T]))

#Throughput
model.Prod = pyo.Var(domain=pyo.NonNegativeReals)
model.Prodcon = pyo.Constraint(expr = model.Prod == model.S['CW',H])

#Objective function defined as maximisation of throughput
model.obj = pyo.Objective(expr = model.Prod + model.SVal - model.OpCost, sense = pyo.maximize)

###############################################################################

##MODELLING: EQUATIONS AND CONSTRAINTS

#Create constraint environment
model.con = pyo.ConstraintList()

#Constraint 1: Only one task per unit at time t
for j in J:
    for t in T:
        eq = 0
        for i in Ij[j]:
            for n in T:
                if n >= (t-tau[i]+1) and n <= t:
                    eq = eq + model.W[i,j,n]
        model.con.add(eq <= 1)
        
#Constraint 2: Unit minimum and maximum capacity
for t in T:
    for j in J:
        for i in Ij[j]:
            model.con.add(model.W[i,j,t]*Betamin[i,j] <= model.B[i,j,t])
            model.con.add(model.B[i,j,t] <= model.W[i,j,t]*Betamax[i,j]) 
    
#Constraint 3: Storage capacity/maximum inventory
model.Scon = pyo.Constraint(K.keys(), T, rule = lambda model, k, t: model.S[k,t] <= Sk_max[k])

#Equation 1: Development of inventory over time (Mass balances)
for k in K.keys():
    eq = K[k]['Ini']
    for t in T:
        for i in Iplus[k]:
            for j in Ji[i]:
                if t >= tauK[(i,k)]: 
                    eq = eq + rho[(i,k)]*model.B[i,j,max(T[T <= t-tauK[(i,k)]])]             
        for i in Iminus[k]:
            eq = eq - xi[(i,k)]*sum([model.B[i,j,t] for j in Ji[i]])
        model.con.add(model.S[k,t] == eq)
        eq = model.S[k,t] 

#Equation 2: Development of M over time
for j in J:
    eq = 0
    for t in T:
        eq = eq + sum([model.B[i,j,t] for i in Ij[j]])
        for i in Ij[j]:
                for k in Kplus[i]:
                    if t >= tauK[(i,k)]:
                        eq = eq - rho[(i,k)]*model.B[i,j,max(T[T <= t-tauK[(i,k)]])]
        model.con.add(model.M[j,t] == eq)
        eq = model.M[j,t]

#Additional constraint: No active units at the end of the scheduling horizon
model.tc = pyo.Constraint(J, rule = lambda model, j: model.M[j,H] == 0)

###############################################################################

##CIP EQUATIONS AND CONSTRAINTS

#Gather maximum health and task health use
etamax = {j:JI_union[(j,i)]['etamax'] for (j,i) in JI_union}
theta = {(j,i):JI_union[(j,i)]['theta'] for (j,i) in JI_union}

#Define health variables
model.H = pyo.Var(J, T, domain=pyo.NonNegativeIntegers)
model.Hp = pyo.Var(J, T, domain=pyo.NonNegativeIntegers)

#Health: Bounds and relationships
for j in J:
    eq = etamax[j]
    for t in T:
        model.con.add(model.H[j,t] <= etamax[j])
        model.con.add(model.H[j,t] >= 0)
        model.con.add(model.Hp[j,t] <= (etamax[j])*model.W['aCIP',j,t])
        if t >= tgap:
            model.con.add(model.Hp[j,t] <= etamax[j] - model.H[j,t-tgap])
            model.con.add(model.Hp[j,t] >= (etamax[j])*model.W['aCIP',j,t] - model.H[j,t-tgap])
        eq = eq - sum([model.W[i,j,t]*theta[j,i] for i in Ij[j]]) + model.Hp[j,t]
        model.con.add(model.H[j,t] == eq)
        eq = model.H[j,t]
        
#Only 1 CIP at a time
for t in T:
    eq = 0
    for n in T:
        if n >= (t-tau['aCIP']+1) and n <= t:
            eq = eq + sum([model.W['aCIP',j,n] for j in J])
    model.con.add(eq <= 1)

###############################################################################

##SOLVE MODEL AND VISUALISE

#Solve the model with PYOMO optimisation
SolverFactory('cplex').solve(model,options_string='mipgap=0.0001').write()

#Visualise solution in Gantt chart
plt.figure(figsize=(15,3))

#Gap between bars
bargap = 1/1000*H/60
#Initialisation
marks = []
lbls = []
idp = 1
Jsort = ['MillMash 1','MillMash 2','Lauter Tun 1','Lauter Tun 2','Wort Kettle','WhirlCool']
#Plotting over units and tasks - some formatting is performed for tasks relating to beer types
for j in Jsort:
    idp = idp - 1
    for i in sorted(Ij[j]):
        idp = idp - 1
        #Marks and titles
        marks.append(idp)
        if j == 'Wort Kettle' and i == 'aCIP' or j == 'WhirlCool' and i == 'aCIP':
            lbls.append("{0:s} (CIP)".format(j))
        else:
            lbls.append("{0:s}".format(j))
        for t in T:
            if model.W[i,j,t]() > 0:
                #Gantt chart bar
                if i == 'aCIP':
                    plt.plot([t/60+bargap,t/60+tau[i]/60-bargap], [idp,idp],alpha=.5,color='r', lw=15, solid_capstyle='butt')
                else:
                    plt.plot([t/60+bargap,t/60+tau[i]/60-bargap], [idp,idp],alpha=.5,color='c', lw=15, solid_capstyle='butt')
                #Gantt chart text
                txt = "{0:.0f}".format(model.B[i,j,t]())
                plt.text(t/60+tau[i]/60/2, idp, txt, color='k', weight='bold', ha='center', va='center')
#Axis formatting
plt.xlim(0,H/60)
plt.xlabel("Time [h]", fontweight='bold')
plt.ylabel("Units", fontweight='bold')
plt.gca().set_yticks(marks)
plt.gca().set_yticklabels(lbls);
# %%
