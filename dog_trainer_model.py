import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import sin, cos



def trainer_model(y,t):
    dxdt = -5*sin(t)  # Geschwindigkeit in x-Richtung
    dydt = 5*cos(t)  # Geschwindigkeit in y-Richtung
    return [dxdt,dydt]


# Anfangsbedingungen Trainer
x0=5
y0=0
InitialConditions_trainer=[x0, y0]

# Zeitschritte
t = np.linspace(0, 60, 501)

# Das DGE lösen
trainer = odeint(trainer_model, InitialConditions_trainer, t)


#Model eines Hundes mittels Vektorprojektion
def dog_model(y,t):
    v_trainer = np.array([-5 * sin(t), 5 * cos(t)])
    d = np.array([y[0] - 5 * cos(t), y[1] - 5 * sin(t)])
    v_d = (np.dot(v_trainer, d) / np.dot(d,d)) * d
    return v_d


# Anfangsbedingungen Hund
x0=10
y0=0
InitialConditions_dog=[x0,y0]


# Das DGE lösen
dog = odeint(dog_model, InitialConditions_dog, t)

# Grafik erstellen
plt.plot(dog[:,0], dog[:,1])
plt.plot(trainer[:,0], trainer[:,1])
plt.xlabel('x-Richtung')
plt.ylabel('y-Richtung')
plt.grid()
plt.show()