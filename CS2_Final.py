# Brewery Case Study - Final Implementation
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
     #General
        'Water':  {'Cap': 100000,'Ini': 100000,'nu':  0},
        'SG':     {'Cap': 10000,'Ini':   0,  'nu':  0},
        'Waste':  {'Cap': 10000,'Ini':   0,  'nu':  0},
        
     #Malt
        'Mp':     {'Cap': 100000,'Ini': 1392.8208,'nu':  0},
        'Mw':     {'Cap': 10000,'Ini': 408.992,'nu':  0},
        'Ms':     {'Cap': 10000,'Ini': 15.3372,'nu':  0},
        'Mo':     {'Cap': 10000,'Ini': 383.43,'nu':  0},
        
     #Pilsner-specific
        'HMp':    {'Cap': 520,  'Ini':   0,  'nu':  -0.01},
        'Wp':     {'Cap': 510,  'Ini':   0,  'nu':  -0.01},
        'BWp':    {'Cap': 440,  'Ini':   0,  'nu':  -0.01},
        'CWp':    {'Cap': 10000,'Ini':   0,  'nu':  0.01},
        
     #Wheat-specific
        'HMw':    {'Cap': 520,  'Ini':   0,  'nu':  -0.01},
        'Ww':     {'Cap': 510,  'Ini':   0,  'nu':  -0.01},
        'BWw':    {'Cap': 440,  'Ini':   0,  'nu':  -0.01},
        'CWw':    {'Cap': 10000,'Ini':   0,  'nu':  0.01},
        
     #Schwarz-bier
        'HMs':    {'Cap': 520,  'Ini':   0,  'nu':  -0.01},
        'Ws':     {'Cap': 510,  'Ini':   0,  'nu':  -0.01},
        'BWs':    {'Cap': 440,  'Ini':   0,  'nu':  -0.01},
        'CWs':    {'Cap': 10000,'Ini':   0,  'nu':  0.01},
        
     #Organic Pilsner
         'HMo':    {'Cap': 520,  'Ini':   0,  'nu':  -0.01},
         'Wo':     {'Cap': 510,  'Ini':   0,  'nu':  -0.01},
         'BWo':    {'Cap': 440,  'Ini':   0,  'nu':  -0.01},
         'CWo':    {'Cap': 10000,'Ini':   0,  'nu':  0.01},
         
     #CIP
         'CIPin':  {'Cap': 10000,'Ini': 10000,'nu':  0},
         'CIPout': {'Cap': 10000,'Ini':   0,  'nu':  0},
    }

#State-to-Task nodes with feed amount/stoichiometry (as KtI for material to task)
KtI = {
       #Pilsner
        ('Mp',    'MillMashing - P'):   {'xi': 0.25},
        ('Water', 'MillMashing - P'):   {'xi': 0.75},
        ('HMp',   'Lautering - P'):     {'xi': 0.75},
        ('Water', 'Lautering - P'):     {'xi': 0.25},
        ('Wp',    'Boiling - P'):       {'xi': 1.0},
        ('BWp',   'WhirlCooling - P'):  {'xi': 1.0},
        
        #Wheat beer
        ('Mp',  'MillMashing - W'):     {'xi': 0.09},
        ('Mw',  'MillMashing - W'):     {'xi': 0.16},
        ('Water', 'MillMashing - W'):   {'xi': 0.75},
        ('HMw',    'Lautering - W'):    {'xi': 0.75},
        ('Water', 'Lautering - W'):     {'xi': 0.25},
        ('Ww',     'Boiling - W'):      {'xi': 1.0},
        ('BWw',    'WhirlCooling - W'): {'xi': 1.0},
        
        #Schwarz-bier
        ('Mp',  'MillMashing - xS'):   {'xi': 0.235},
        ('Ms',  'MillMashing - xS'):   {'xi': 0.015},
        ('Water', 'MillMashing - xS'): {'xi': 0.75},
        ('HMs',    'Lautering - xS'):     {'xi': 0.75},
        ('Water', 'Lautering - xS'):     {'xi': 0.25},
        ('Ws',     'Boiling - xS'):       {'xi': 1.0},
        ('BWs',    'WhirlCooling - xS'):  {'xi': 1.0},
        
        #Organic
        ('Mo',  'MillMashing - zO'):   {'xi': 0.25},
        ('Water', 'MillMashing - zO'): {'xi': 0.75},
        ('HMo',    'Lautering - zO'):     {'xi': 0.75},
        ('Water', 'Lautering - zO'):     {'xi': 0.25},
        ('Wo',     'Boiling - zO'):       {'xi': 1.0},
        ('BWo',    'WhirlCooling - zO'):  {'xi': 1.0},
        
        #CIP
        ('CIPin', 'aCIP'):           {'xi': 1.0},
    }

#Task-to-State nodes with task processing time and conversion coefficient
ItK = {
       #Pilsner
       ('MillMashing - P', 'HMp'):    {'tau': 135,  'rho': 0.9025},
       ('MillMashing - P', 'Waste'):  {'tau': 135,  'rho': 0.0975},
       ('Lautering - P', 'Wp'):       {'tau': 90,   'rho': 0.7315},
       ('Lautering - P', 'SG'):       {'tau': 90,   'rho': 0.23},
       ('Lautering - P', 'Waste'):    {'tau': 90,   'rho': 0.0385},
       ('Boiling - P', 'BWp') :       {'tau': 105,  'rho': 0.857375},
       ('Boiling - P', 'Waste') :     {'tau': 105,  'rho': 0.142625},
       ('WhirlCooling - P', 'CWp'):   {'tau': 60,   'rho': 0.9025},
       ('WhirlCooling - P', 'Waste'): {'tau': 60,   'rho': 0.0975},
       
       #Wheat beer
        ('MillMashing - W', 'HMw'):    {'tau': 135,  'rho': 0.9025},
        ('MillMashing - W', 'Waste'):  {'tau': 135,  'rho': 0.0975},
        ('Lautering - W', 'Ww'):       {'tau': 90,   'rho': 0.7315},
        ('Lautering - W', 'SG'):       {'tau': 90,   'rho': 0.23},
        ('Lautering - W', 'Waste'):    {'tau': 90,   'rho': 0.0385},
        ('Boiling - W', 'BWw') :       {'tau': 105,  'rho': 0.857375},
        ('Boiling - W', 'Waste') :     {'tau': 105,  'rho': 0.142625},
        ('WhirlCooling - W', 'CWw'):   {'tau': 60,   'rho': 0.9025},
        ('WhirlCooling - W', 'Waste'): {'tau': 60,   'rho': 0.0975},
        
        #Schwarz-bier
         ('MillMashing - xS', 'HMs'):    {'tau': 135,  'rho': 0.9025},
         ('MillMashing - xS', 'Waste'):  {'tau': 135,  'rho': 0.0975},
         ('Lautering - xS', 'Ws'):       {'tau': 90,   'rho': 0.7315},
         ('Lautering - xS', 'SG'):       {'tau': 90,   'rho': 0.23},
         ('Lautering - xS', 'Waste'):    {'tau': 90,   'rho': 0.0385},
         ('Boiling - xS', 'BWs') :       {'tau': 105,'rho': 0.857375},
         ('Boiling - xS', 'Waste') :     {'tau': 105,'rho': 0.142625},
         ('WhirlCooling - xS', 'CWs'):   {'tau': 60, 'rho': 0.9025},
         ('WhirlCooling - xS', 'Waste'): {'tau': 60, 'rho': 0.0975},
         
         #Organic
          ('MillMashing - zO', 'HMo'):    {'tau': 135,  'rho': 0.9025},
          ('MillMashing - zO', 'Waste'):  {'tau': 135,  'rho': 0.0975},
          ('Lautering - zO', 'Wo'):       {'tau': 90,   'rho': 0.7315},
          ('Lautering - zO', 'SG'):       {'tau': 90,   'rho': 0.23},
          ('Lautering - zO', 'Waste'):    {'tau': 90,   'rho': 0.0385},
          ('Boiling - zO', 'BWo') :       {'tau': 105,  'rho': 0.857375},
          ('Boiling - zO', 'Waste') :     {'tau': 105,  'rho': 0.142625},
          ('WhirlCooling - zO', 'CWo'):   {'tau': 60,   'rho': 0.9025},
          ('WhirlCooling - zO', 'Waste'): {'tau': 60,   'rho': 0.0975},
          
          #CIP
          ('aCIP', 'CIPout'):          {'tau': 90,   'rho': 1},
    }

#Units able to perform specific tasks node (as JI_union) with capacity
#and gamma of performing task
JI_union = {
        #Pilsner
        ('MillMash 1', 'MillMashing - P'):   {'Betamin': 1, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('MillMash 2', 'MillMashing - P'):   {'Betamin': 1, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 1', 'Lautering - P'):   {'Betamin': 1, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 2', 'Lautering - P'):   {'Betamin': 1, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Wort Kettle', 'Boiling - P'):      {'Betamin': 1, 'Betamax': 450, 'gamma': 0.01, 'etamax':3,'theta':1},
        ('WhirlCool', 'WhirlCooling - P'):   {'Betamin': 1, 'Betamax': 411, 'gamma': 0.01, 'etamax':3,'theta':1},
        
        #Wheat beer
        ('MillMash 1', 'MillMashing - W'):   {'Betamin': 1, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('MillMash 2', 'MillMashing - W'):   {'Betamin': 1, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 1', 'Lautering - W'):   {'Betamin': 1, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 2', 'Lautering - W'):   {'Betamin': 1, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Wort Kettle', 'Boiling - W'):      {'Betamin': 1, 'Betamax': 450, 'gamma': 0.01, 'etamax':3,'theta':1},
        ('WhirlCool', 'WhirlCooling - W'):   {'Betamin': 1, 'Betamax': 411, 'gamma': 0.01, 'etamax':3,'theta':1},
        
        #Schwarz-bier
        ('MillMash 1', 'MillMashing - xS'):   {'Betamin': 1, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('MillMash 2', 'MillMashing - xS'):   {'Betamin': 1, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 1', 'Lautering - xS'):   {'Betamin': 1, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 2', 'Lautering - xS'):   {'Betamin': 1, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Wort Kettle', 'Boiling - xS'):      {'Betamin': 1, 'Betamax': 450, 'gamma': 0.01, 'etamax':3,'theta':1},
        ('WhirlCool', 'WhirlCooling - xS'):   {'Betamin': 1, 'Betamax': 411, 'gamma': 0.01, 'etamax':3,'theta':1},
        
        #Organic
        ('MillMash 1', 'MillMashing - zO'):   {'Betamin': 1, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('MillMash 2', 'MillMashing - zO'):   {'Betamin': 1, 'Betamax': 273, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 1', 'Lautering - zO'):   {'Betamin': 1, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Lauter Tun 2', 'Lautering - zO'):   {'Betamin': 1, 'Betamax': 328, 'gamma': 0.01, 'etamax':30,'theta':1},
        ('Wort Kettle', 'Boiling - zO'):      {'Betamin': 1, 'Betamax': 450, 'gamma': 0.01, 'etamax':3,'theta':1},
        ('WhirlCool', 'WhirlCooling - zO'):   {'Betamin': 1, 'Betamax': 411, 'gamma': 0.01, 'etamax':3,'theta':1},
        
        #CIP
        ('MillMash 1', 'aCIP'):             {'Betamin': 1, 'Betamax': 1,  'gamma': 0.01, 'etamax':30,'theta':0},
        ('MillMash 2', 'aCIP'):               {'Betamin': 1, 'Betamax': 1,  'gamma': 0.01, 'etamax':30,'theta':0},
        ('Lauter Tun 1', 'aCIP'):             {'Betamin': 1, 'Betamax': 1,  'gamma': 0.01, 'etamax':30,'theta':0},
        ('Lauter Tun 2', 'aCIP'):               {'Betamin': 1, 'Betamax': 1,  'gamma': 0.01, 'etamax':30,'theta':0},
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
H = 58*60
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
model.Prodcon = pyo.Constraint(expr = model.Prod == 
                               model.S['CWp',H] + model.S['CWw',H] 
                               + model.S['CWs',H] + model.S['CWo',H])

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
model.Hm = pyo.Var(J, T, domain=pyo.NonNegativeIntegers)

#Define changeover variables
model.Y = pyo.Var(I,I,J,T, domain=pyo.Boolean)
model.X = pyo.Var(I,J,T, domain=pyo.Boolean)

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
            model.con.add(model.Hm[j,t] <= model.H[j,t-tgap])
            for i in Ij[j]:
                if i.endswith('- xS'):
                    for im in Ij[j]:
                        model.con.add(model.Hm[j,t] >= model.H[j,t-tgap] - etamax[j] * (1-model.Y[i,im,j,t]) )
                        model.con.add(model.Hm[j,t] <= etamax[j]* model.Y[i,im,j,t])
        eq = eq - sum([model.W[i,j,t]*theta[j,i] for i in Ij[j]]) + model.Hp[j,t] - model.Hm[j,t]
        model.con.add(model.H[j,t] == eq)
        eq = model.H[j,t]

#Bounds and relationships
for j in J:
    for i in Ij[j]:
        if i.endswith('- xS'):
            eq = 1
        else:
            eq = 0
        for t in T:
            model.con.add(sum([model.X[iw,j,t] for iw in Ij[j]]) == 1)
            model.con.add(sum([model.Y[i,im,j,t] for i in Ij[j] for im in Ij[j] for t in T]) >= 1)
            eqY1 = 0
            eqY2 = 0
            if t >= tau[i]:
                model.con.add(sum([model.W[i,j,z] for z in tgap*np.array(range(int((t-tau[i])/tgap),int(t/tgap)))]) <= model.X[i,j,t])
            for im in Ij[j]:
                if i != im and t >= 3*tau['MillMashing - P']-tgap:
                    eqY1 = eqY1 + model.Y[im,i,j,t-tgap]
                    eqY2 = eqY2 + model.Y[i,im,j,t-tgap]
                elif i == im:
                    model.con.add(model.Y[i,im,j,t] == 0)
            eq = eq + eqY1 - eqY2
            model.con.add(model.X[i,j,t] == eq)
            eq = model.X[i,j,t]

for j in J:
    for im in Ij[j]:
        if im.endswith('- zO'):
            for t in T:
                for i in Ij[j]:
                    model.con.add(etamax[j]*model.Y[i,im,j,t] <= model.H[j,t])

#Only 1 CIP at a time
for t in T:
    eq = 0
    for n in T:
        if n >= (t-tau['aCIP']+1) and n <= t:
            eq = eq + sum([model.W['aCIP',j,n] for j in J])
    model.con.add(eq <= 1)
    
#Throughput
model.CO = pyo.Var(domain=pyo.NonNegativeReals)
model.COcon = pyo.Constraint(expr = model.CO == 0.1*sum([model.Y[i,im,j,t] for i in I for im in I for j in Ji[i] for t in T]))

#Objective function defined as maximisation of throughput
model.obj = pyo.Objective(expr = model.Prod + model.SVal - model.OpCost - model.CO, sense = pyo.maximize)

###############################################################################

##SOLVE MODEL AND VISUALISE

#Solve the model with PYOMO optimisation
solver = SolverFactory('cplex')
solver.options['mipgap'] = 0.0001
solver.options['timelimit'] = 21600

solver.solve(model,tee=True)

#Visualise solution in Gantt chart
plt.figure(figsize=(15,7))

#Gap between bars
bargap = 1/1000*H/60
#Initialisation
marks = []
lbls = []
idp = 1
Jsort = ['MillMash 1','MillMash 2','Lauter Tun 1','Lauter Tun 2','Wort Kettle','WhirlCool']
for j in Jsort:
    idp = idp - 1
    idBeerType = 0
    for i in sorted(Ij[j]):
        idp = idp - 1
        idBeerType = idBeerType + 1
        #Marks and titles
        marks.append(idp)
        if idBeerType == 1:
            lbls.append("{0:s} (P)".format(j))
        elif idBeerType == 2:
            lbls.append("{0:s} (W)".format(j))
        elif idBeerType == 3:
            lbls.append("{0:s} (S)".format(j))
        elif idBeerType == 4:
            lbls.append("{0:s} (O)".format(j))
        elif idBeerType == 5:
            lbls.append("{0:s} (CIP)".format(j))
        for t in T:
            if model.W[i,j,t]() > 0.1:
                #Gantt chart bar
                if idBeerType == 1:
                    plt.plot([t/60+bargap,t/60+tau[i]/60-bargap], [idp,idp],alpha=.5,color='c', lw=15, solid_capstyle='butt')
                elif idBeerType == 2:
                    plt.plot([t/60+bargap,t/60+tau[i]/60-bargap], [idp,idp],alpha=.5,color='m', lw=15, solid_capstyle='butt')
                elif idBeerType == 3:
                    plt.plot([t/60+bargap,t/60+tau[i]/60-bargap], [idp,idp],alpha=.5,color='darkorange', lw=15, solid_capstyle='butt')
                elif idBeerType == 4:
                    plt.plot([t/60+bargap,t/60+tau[i]/60-bargap], [idp,idp],alpha=.5,color='lime', lw=15, solid_capstyle='butt')
                elif idBeerType == 5:
                    plt.plot([t/60+bargap,t/60+tau[i]/60-bargap], [idp,idp],alpha=.5,color='r', lw=15, solid_capstyle='butt')
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
