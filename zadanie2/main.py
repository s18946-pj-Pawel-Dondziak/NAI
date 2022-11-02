"""
This app uses fuzzy control system, it lets you see if your meat is ready
temperature, mass and time that is left are needed
To properly run the app install : scikit-fuzzy and matplotlib
You can use these commands:
pip install scikit-fuzzy
pip install matplotlib
"""

from matplotlib import pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# definition of input variables
# temperature is given in celsius, mass in kilograms, time in minutes
temperature = ctrl.Antecedent(np.arange(100, 301, 10), 'temperature')
mass = ctrl.Antecedent(np.arange(0.5, 4, 0.1), 'mass')
time = ctrl.Antecedent(np.arange(10, 101, 10), 'time')

# definition of output variables
preparationType = ctrl.Consequent(np.arange(0, 121, 1), 'preparationType')

# Auto-membership function population is possible with .automf(3, 5, or 7)
temperature.automf(3)
mass.automf(3)
time.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
preparationType['NotDone'] = fuzz.trimf(preparationType.universe, [0, 0, 55])
preparationType['Rare'] = fuzz.trimf(preparationType.universe, [50, 55, 65])
preparationType['MediumRare'] = fuzz.trimf(preparationType.universe, [60, 65, 75])
preparationType['Medium'] = fuzz.trimf(preparationType.universe, [70, 75, 85])
preparationType['MediumWell'] = fuzz.trimf(preparationType.universe, [80, 85, 95])
preparationType['WellDone'] = fuzz.trimf(preparationType.universe, [90, 95, 105])
preparationType['OverDone'] = fuzz.trimf(preparationType.universe, [100, 120, 120])


temperature.view()
mass.view()
time.view()
preparationType.view()

# if we have short time left, the temperature is high and mass is poor or medium, the meat is overcooked
rule1 = ctrl.Rule(time['poor'] & temperature['good'], preparationType['OverDone'])

rule2 = ctrl.Rule(time['poor'] & (temperature['average'] | temperature['good'])
                  & (mass['average'] | mass['good']), preparationType['WellDone'])

rule3 = ctrl.Rule(time['average'] & temperature['average'] & mass['average'], preparationType['MediumWell'])

rule4 = ctrl.Rule(time['average'] & temperature['average'], preparationType['Medium'])

rule5 = ctrl.Rule(time['average'] & temperature['average'] & (mass['average'] | mass['poor'])
                  | (time['average'] & temperature['good'] & mass['good']), preparationType['MediumRare'])

rule6 = ctrl.Rule((time['average'] & (mass['average'] | mass['poor']))
                  | (time['average'] & temperature['average']), preparationType['Rare'])

# if there is much time left, meat is not done yet
rule7 = ctrl.Rule(time['good'], preparationType['NotDone'])

prepType_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])

preparation = ctrl.ControlSystemSimulation(prepType_ctrl)


# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
preparation.input['temperature'] = 5
preparation.input['mass'] = 5
preparation.input['time'] = 5

# Crunch the numbers
preparation.compute()

print(preparation.output['preparationType'])
preparationType.view(sim=preparation)

plt.show()
