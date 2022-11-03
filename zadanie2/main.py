"""
Jakub Świderski s19443
Paweł Dondziak s18946
This app uses fuzzy control system, to determine the process of meat preparation in electric oven.
The program takes three numeric variables:
  - Temperature
  - Mass
  - Time
To properly run the app install : scikit-fuzzy and matplotlib
You can use these commands:
pip install scikit-fuzzy
pip install matplotlib

skfuzzy documentation : https://scikit-fuzzy.readthedocs.io/en/latest/
"""

from matplotlib import pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# definition of input variables
# temperature is given in celsius, mass in kilograms, time in minutes
temperature = ctrl.Antecedent(np.arange(100, 320, 1), 'temperature')
mass = ctrl.Antecedent(np.arange(1, 3.1, 0.1), 'mass')
time = ctrl.Antecedent(np.arange(0, 120, 1), 'time')

# definition of output variables
preparationType = ctrl.Consequent(np.arange(0, 101, 1), 'preparationType')

# Auto-membership function population is possible with .automf(3, 5, or 7)
temperature.automf(3)
mass.automf(3)
time.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
preparationType['Raw'] = fuzz.trimf(preparationType.universe, [0, 0, 50])
preparationType['Medium'] = fuzz.trimf(preparationType.universe, [40, 65, 85])
preparationType['WellDone'] = fuzz.trimf(preparationType.universe, [70, 100, 100])


temperature.view()
mass.view()
time.view()
preparationType.view()

# if we have short time left, the temperature is high and mass is poor or medium, the meat is overcooked

rule1 = ctrl.Rule(time['poor'], preparationType['Raw'])
rule2 = ctrl.Rule(time['average'] & mass['good'], preparationType['Raw'])
rule3 = ctrl.Rule(time['average'] & temperature['good'], preparationType['Medium'])
rule4 = ctrl.Rule(time['average'], preparationType['Medium'])
rule5 = ctrl.Rule(time['good'] & temperature['poor'], preparationType['Medium'])
rule6 = ctrl.Rule(time['good'] & mass['poor'], preparationType['Medium'])
rule7 = ctrl.Rule(time['average'] & (temperature['good'] | mass['poor']), preparationType['WellDone'])
rule8 = ctrl.Rule(time['good'], preparationType['WellDone'])

prepType_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])

preparation = ctrl.ControlSystemSimulation(prepType_ctrl)


# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
preparation.input['temperature'] = 280
preparation.input['mass'] = 1.8
preparation.input['time'] = 100

# Crunch the numbers
preparation.compute()

print(preparation.output['preparationType'])
preparationType.view(sim=preparation)

plt.show()