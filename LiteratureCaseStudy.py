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

###############################################################################
##DICTIONARIES are created to carry data in a transparent and general manner
#States (as K for material)
K = {
        'FA':   {'Cap': 500, 'Ini': 500, 'nu':  0},
        'FB':   {'Cap': 500, 'Ini': 500, 'nu':  0},
        'FC':   {'Cap': 500, 'Ini': 500, 'nu':  0},
        'HotA': {'Cap': 100, 'Ini':   0, 'nu': -1},
        'AB':   {'Cap': 200, 'Ini':   0, 'nu': -1},
        'BC':   {'Cap': 150, 'Ini':   0, 'nu': -1},
        'E':    {'Cap': 100, 'Ini':   0, 'nu': -1},
        'P1':   {'Cap': 500, 'Ini':   0, 'nu': 10},
        'P2':   {'Cap': 500, 'Ini':   0, 'nu': 10},
    }

#State-to-Task nodes with feed amount/stoichiometry (as KtI for material to task)
KtI = {
        ('FA',   'Heating'):    {'xi': 1.0},
        ('FB',   'R1'):         {'xi': 0.5},
        ('FC',   'R1'):         {'xi': 0.5},
        ('FC',   'R3'):         {'xi': 0.2},
        ('HotA', 'R2'):         {'xi': 0.4},
        ('AB',   'R3'):         {'xi': 0.8},
        ('BC',   'R2'):         {'xi': 0.6},
        ('E',    'Separation'): {'xi': 1.0},
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
        ('Heater', 'Heating'):    {'Betamin': 0, 'Betamax': 100, 'gamma': 0.01},
        ('Reactor 1', 'R1'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 0.01},
        ('Reactor 1', 'R2'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 0.01},
        ('Reactor 1', 'R3'):       {'Betamin': 0, 'Betamax':  80, 'gamma': 0.01},
        ('Reactor 2', 'R1'):       {'Betamin': 0, 'Betamax':  50, 'gamma': 0.01},
        ('Reactor 2', 'R2'):       {'Betamin': 0, 'Betamax':  50, 'gamma': 0.01},
        ('Reactor 2', 'R3'):       {'Betamin': 0, 'Betamax':  50, 'gamma': 0.01},
        ('Still', 'Separation'):  {'Betamin': 0, 'Betamax': 200, 'gamma': 0.01},
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
model.SValcon = pyo.Constraint(expr = model.SVal == sum([K[k]['nu']*model.S[k,H] for k in K]))

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

#Lifted variable handling: M is inventory of unit j at time t
model.M = pyo.Var(J, T, domain=pyo.NonNegativeReals)

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

#Equation 2 and 3: Development of W and B over time (using M)
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

##SOLVE MODEL AND VISUALISE

#Solve the model with PYOMO optimisation
SolverFactory('cplex').solve(model).write()

#Visualise solution in Gantt chart
plt.figure(figsize=(10,3))

bargap = 1/500*H
marks = []
lbls = []
idp = 1
for j in sorted(J):
    idp = idp - 1
    for i in sorted(Ij[j]):
        idp = idp - 1
        marks.append(idp)
        lbls.append("{0:s} ({1:s})".format(j,i))
        for t in T:
            if model.W[i,j,t]() > 0:
                plt.plot([t+bargap,t+tau[i]-bargap], [idp,idp],alpha=.5,color='c', lw=15, solid_capstyle='butt')
                txt = "{0:.2f}".format(model.B[i,j,t]())
                plt.text(t+tau[i]/2, idp, txt, color='k', weight='bold', ha='center', va='center')
plt.xlim(0,H)
plt.xlabel("Time [h]", fontweight='bold')
plt.ylabel("Units (Tasks)", fontweight='bold')
plt.gca().set_yticks(marks)
plt.gca().set_yticklabels(lbls);

###############################################################################

# VISUALISATION OF DOCUMENTED EXAMPLE DATA

# Setup literature data
Wart = np.zeros((len(I),len(J),len(T)))
Bart = np.zeros((len(I),len(J),len(T)))
#Heater
Wart[0,0,1] = 1; Bart[0,0,1] = 52;
Wart[0,0,3] = 1; Bart[0,0,3] = 32;
Wart[0,0,7] = 1; Bart[0,0,7] = 52;
#Reactor 1
Wart[1,1,0] = 1; Bart[1,1,0] = 80;
Wart[2,1,2] = 1; Bart[2,1,2] = 80;
Wart[2,1,4] = 1; Bart[2,1,4] = 80;
Wart[1,1,6] = 1; Bart[1,1,6] = 78;
Wart[2,1,8] = 1; Bart[2,1,8] = 80;
#Reactor 2
Wart[1,2,0] = 1; Bart[1,2,0] = 46;
Wart[2,2,2] = 1; Bart[2,2,2] = 50;
Wart[3,2,4] = 1; Bart[3,2,4] = 50;
Wart[3,2,5] = 1; Bart[3,2,5] = 48;
Wart[3,2,6] = 1; Bart[3,2,6] = 50;
Wart[3,2,7] = 1; Bart[3,2,7] = 17;
Wart[2,2,8] = 1; Bart[2,2,8] = 50;
#Still
Wart[4,3,5] = 1; Bart[4,3,5] = 50;
Wart[4,3,8] = 1; Bart[4,3,8] = 114;

#Visualise solution in Gantt chart
plt.figure(figsize=(10,3))

bargap = 1/500*H
marks = []
lbls = []
idp = 1
iart = 0; jart = 0;
for j in sorted(J):
    if j == 'Heater':
        iart = 0
    elif j == 'Reactor 1' or j == 'Reactor 2':
        iart = 1
    elif j == 'Still':
        iart = 4
    idp = idp - 1
    for i in Ij[j]:
        idp = idp - 1
        marks.append(idp)
        lbls.append("{0:s} ({1:s})".format(j,i))
        for t in T:
            if Wart[iart,jart,t] > 0:
                plt.plot([t+bargap,t+tau[i]-bargap], [idp,idp],alpha=.5,color='c', lw=15, solid_capstyle='butt')
                txt = "{0:.2f}".format(Bart[iart,jart,t])
                plt.text(t+tau[i]/2, idp, txt, color='k', weight='bold', ha='center', va='center')
        iart = iart + 1
    jart = jart + 1
plt.xlim(0,H)
plt.xlabel("Time [h]", fontweight='bold')
plt.ylabel("Units (Tasks)", fontweight='bold')
plt.gca().set_yticks(marks)
plt.gca().set_yticklabels(lbls);