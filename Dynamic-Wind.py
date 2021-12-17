#-------------------------------------------------------------------------    
#1. Importando módulos
#------------------------------------------------------------------------- 

import numpy as np
from numpy.random import default_rng
import scipy.linalg as sc
from scipy.optimize import minimize,Bounds
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from pandas import ExcelWriter

#obs.: Importar a matriz de rigidez do sistem (K)

#-------------------------------------------------------------------------        
#2. Modelagem do vento
#------------------------------------------------------------------------- 

#9.1 Dados de Entrada
v0    = 43    # Velocidade básica do vento em m/s (NBR 6123 - Balneário Camboriú)
zref  = 10    # Altura de referência em m
z0    = 0.07  # Comprimento de rugosidade em m  categoria II (NBR 6123 - zonas costeiras planas)
s1    = 1     # Fator de correção topográfico (NBR 6123 - Terrenos planos ou fracamente acidentados)
s3    = 1     # Fator de correção Estatístico (NBR 6123 - Ocupação residencial) 
b     = 1     # categoria II, Classe C (NBR 6123 - zonas costeiras planas, Maior dimensão excede 50m)
p     = 0.1   # categoria II, Classe C (NBR 6123 - zonas costeiras planas, Maior dimensão excede 50m)
fr    = 0.95  # categoria II, Classe C (NBR 6123 - zonas costeiras planas, Maior dimensão excede 50m)
Ca    = 1.45  # Coeficiente de arrasto (NBR 6123 - Figura 4 - l1=l2=15m, h=99,75m )

#Vetor de alturas
z      = np.linspace(2.85,99.75,35)

#Velocidade média para plotagem do gráfico
vm = np.zeros((len(z),1))
s2 = np.zeros((len(z),1))

for i in range(len(z)):
    s2[i] = b*fr*((z[i]/10)**p)

for i in range(len(vm)):
    vm[i] = s1*s2[i]*s3*v0
    
#9.2 Shinozuka e Jan
# duraçao : tempo de duração do sinal
# dt : Passo temporal
dur     = 300    #seg
dt      = 0.0005  #seg
tf      = int(dur/dt)
f       = np.linspace(10**-5,10,tf) # Intervalo de variação da frequência
df      = f[1]-f[0]
t       = np.linspace(0,dur,tf)
Sa      = np.zeros(tf)
Sb      = np.zeros(tf)
Sc      = np.zeros(tf)
Sd      = np.zeros(tf)
rng1    = np.random.default_rng(seed=27)
phasea  = rng1.random(tf)*2*np.pi
rng2    = np.random.default_rng(seed=54)
phaseb  = rng2.random(tf)*2*np.pi
rng3    = np.random.default_rng(seed=86)
phasec  = rng3.random(tf)*2*np.pi
rng4    = np.random.default_rng(seed=43)
phased  = rng4.random(tf)*2*np.pi
deltaVa = np.zeros(tf)
deltaVb = np.zeros(tf)
deltaVc = np.zeros(tf)
deltaVd = np.zeros(tf)

v10         = b*fr*((10/10)**p)*v0*s1*s3
u_asterisco = 0.4 * ((v10)/(np.log(zref/z0)))

#Davenport
L1          = 1200
n1          = (f*L1)/v10
fSw         = (4*(n1**2))/((1+(n1**2))**(4/3))
Sw          = fSw * ((u_asterisco**2)/(f))

for i in range(tf):
    Sa = np.sqrt(2*Sw*df)*np.cos(2*np.pi*f*t[i] + phasea)
    deltaVa[i] = sum(Sa)   
    
for i in range(tf):
    Sb = np.sqrt(2*Sw*df)*np.cos(2*np.pi*f*t[i] + phaseb)
    deltaVb[i] = sum(Sb)   
    
for i in range(tf):
    Sc = np.sqrt(2*Sw*df)*np.cos(2*np.pi*f*t[i] + phasec)
    deltaVc[i] = sum(Sc)
    
for i in range(tf):
    Sd = np.sqrt(2*Sw*df)*np.cos(2*np.pi*f*t[i] + phased)
    deltaVd[i] = sum(Sd)


# deltaVa = deltavA
# deltaVb = deltavB
# deltaVc = deltavC
# deltaVd = deltavD


#9.3 Vetor de Velocidades Final

# Vetores vazios discretizados no tempo para análise dinâmica 
vf       = np.zeros((len(K),tf)) #Vetor de velocidades final discretizado no tempo
vm1      = np.zeros((len(K),1))  #Vetor de velocidades médias a ser repetido no tempo
vm_array = np.zeros((len(K),tf)) #Vetor de velocidades médias discretizado no tempo
deltaV   = np.zeros((len(K),tf)) #Vetor de velocidades flutuantes discretizado no tempo

#Identificando os graus de liberdade a serem carregados
ncarregados = list(np.arange(5,142,4))  #Nós carregados
glcarregados = []                      #Lista vazia para gl carregados
for i in range(len(ncarregados)):
    temp  = (ncarregados[i]*3)-2       #regra de identificação do gl
    temp -= 13                         #Indexização do gl, removendo gl restringidos
    glcarregados.append(temp)

#Aplicando as velocidades médias ao vetor discretizado no tempo
for i in range(len(glcarregados)):
    aux1 = glcarregados[i]
    aux2 = z[i] 
    vm1[aux1] = s1*b*fr*((aux2/10)**p)*s3*v0    
for i in range(tf):
    vm_array[:,i] = vm1[:,0]
    
# Determinando o Comprimento de Correlação (Blessmann, 1995 adaptado por MIGUEL et al., 2012)
htotal = 99.75
ah = L2 = 1.6*htotal + 22.1   #Horizontal
bv = L3 = 0.93*htotal + 29.3  #Vertical

# Determinando a coordenada Horizontal do pórtico no centro do espaço de correlação
x_coor = ah/2

# Determinando as velocidades flutuantes nos graus de liberdade horizontais (Riera e Ambrosini, 1992 adaptado por MIGUEL et al., 2012)
for i in range(tf):
    for j in range(len(glcarregados)):
        aux3 = glcarregados[j]
        aux4 = z[j]    
        deltaV[aux3,i]  = (deltaVa[i]) + (((deltaVb[i] - deltaVa[i])/ah)*x_coor) + (((deltaVc[i] - deltaVa[i])/bv)*aux4)+(((deltaVd[i] - deltaVc[i] - deltaVb[i] + deltaVa[i])/(ah*bv))*(x_coor*aux4))
     
# Somando as velocidas médias e flutuantes
vf = vm_array + deltaV

#9.4 Vetor de Forças Final
#Definindo a área de influencia de aplicação da Pressão dinâmica para cada Nó
Ae = np.zeros((len(K),1))

#Pavimento tipo
for i in range(len(glcarregados)):
    aux1 = glcarregados[i]    
    Ae[aux1] = 2.85*4.97

#Térreo e Ultimo pavimento
Ae [0,0] = 1.425*4.97
#Ae [408,0] = 1.425*4.97

F = Ca*(0.613*(vf**2))*Ae  #Força de arrasto
