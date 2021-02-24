# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:25:23 2020

@author: Victor Baconnet - victor.baconnet@hotmail.com

All rights reserved
"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#Fonction dont les zéros sont les kL de chaque mode
def f(x):
    return 1./np.cosh(x) - np.cos(x)

def dichotomie(f,g,d):
    """
    Calcule la racine de f dans l'intervalle [g,d]
    Paramètre f doit être une FONCTION. g et d float et g<d
    """
    assert g < d #pour empêcher que les trou du culs comme toi mettent des mauvaises bornes
    c = (g+d)*0.5
    while(abs(f(c)) > 0.00000000001):
        if (f(c)*f(g) < 0.):
            d = c
        else:
            g = c
        c = (g+d)*0.5
    return c


def modes(K,M):
    """
    Donne les vecteurs propres et valeurs propres.
    Retourne un tuple (valeurs propres, vecteurs propres). Paramètres:
        - K : Matrice de raideur carrée
        - M : Matrice de masse carrée
    Les deux matrices doivent être de même dimension
    """
    assert K.shape == M.shape
    return np.linalg.eig(K.dot(np.linalg.inv(M)))



def K_elementaire(E,I,L):
    """
    Matrice de raideur élémentaire pour une poutre en flexion. Paramètres:
        - E : Module de Young
        - I : moment quadratique en Z
        - L : longueur de l'élément = longueur totale/nombre d'éléments
    """
    return E*I/(L*L*L)*np.array([[12.,6.*L,-12.,6.*L],
                                 [6.*L,4.*L*L,-6.*L,2.*L*L],
                                 [-12.,-6.*L,12.,-6.*L],
                                 [6.*L,2.*L*L,-6.*L,4.*L*L]])



def M_elementaire(rho,S,L):
    """
    Matrice de masse élémentaire pour une poutre en flexion. Paramètres:
        - rho : masse volumique
        - S : section de la poutre
        - L : longueur de l'élément = longueur totale/nombre d'éléments
    """
    return rho*S*L/420.*np.array([[156.,22.*L,54.,-13.*L],
                                  [22.*L,4.*L*L,13.*L,-3.*L*L],
                                  [54.,13.*L,156.,-22.*L],
                                  [-13.*L,-3.*L*L,-22.*L,4.*L*L]])



def assembler(N_elts,mat):
    """
    Assemblage des matrices élémentaires dans une grosse matrice (sans prendre
    en compte les conditions limites)
    """
    A = np.zeros((2*N_elts+2,2*N_elts+2))
    for i in range(0,2*N_elts,2):
        A[i:i+4,i:i+4] += mat
    return A


def init_solution(vec):
    """
    Stocke la solution pour chaque mode, avec les solutions nulles d'encastrement aux extrémités.

    vec est la matrice des vecteurs propres calculée avec la fonction modes(). Chaque colonne contient le
    vecteur propre d'un mode donné. Par exemple, le vecteur propre lié au mode 1 est le vecteur vec[:,0],
    et ainsi de suite. Mode i : vec[:,i]

    La fonction retourne juste la matrice vec avec les solutions 0,0 aux extrémités.
    sol[:,i] = [0, 0, vec[:,i], 0, 0]
    """
    sol = np.zeros((2*N_elts+2,len(vec)))
    for i in range(len(vec)):
        sol[2:sol.shape[0]-2,i] = vec[:,i]

    return sol

def find_min(val):
    """
    Trouve l'index du minimum d'un vecteur
    """
    index = 0;
    mini = np.min(val);
    for i in val:
        if abs(mini-i) < 0.00000001:
            return index,mini
        else:
            index += 1
    return index, mini;


def tri_modes(val,vec):
    """
    Trie les modes dans l'ordre croissant car linalg.eig ne rend pas les vp dans l'ordre
    """
    tri_val = np.copy(val); #Va contenir les valeurs propres triées
    tri_vec = np.copy(vec); #Va contenir les vecteurs propres triés, correspondant aux valeur propres
    i=0;
    while (len(val) != 0):
        index, mini = find_min(val);
        tri_val[i] = mini;
        tri_vec[:,i] = vec[:,index];
        val = np.delete(val, index);
        vec = np.delete(vec, index, axis=1);
        i += 1

    return tri_val, tri_vec;

def tracer_mode(x,ax,N,sol):
    """
    Trace le mode N, selon le vecteur x d'abcisse sur le plot ax.
    """
    ax.plot(x,sol[::2,N],label="mode {}".format(N+1));
    ax.set_xlabel('x',fontsize=18)
    ax.set_ylabel('Amplitude',fontsize=18)
    ax.legend(fontsize=18);
    ax.grid(which='both');


N_elts = 50 #Nombre d'éléments
L_tot = 1. #Longueur de la poutre
L = L_tot/float(N_elts) #longueur d'un élément
h = 0.01 #hauteur de la poutre
b = 0.01 #Largeur de la poutre
S = h*b #Section de la poutre
E = 210000000000. #Module Young
I = S*h*h/12.
rho = 7500. #Masse volumique

#Matrices de masse et raideur totales
Mtot = assembler(N_elts, M_elementaire(rho, S, L))
Ktot = assembler(N_elts, K_elementaire(E, I, L))

#On élimine les conditions limites aux extrémités
Mtot = Mtot[2:len(Mtot)-2,2:len(Mtot)-2]
Ktot = Ktot[2:len(Ktot)-2,2:len(Ktot)-2]

val,vec = modes(Ktot,Mtot) #On récupère valeurs propres et vecteurs propres

val = np.sqrt(val) #Valeurs propres --> w² donc on récup les w

val, vec = tri_modes(val, vec); #Pour avoir les modes dans l'ordre
print(val[:5]) #Afficher les 4 premiers modes dans la console
print(val[:5]/(2.*np.pi))
sol = init_solution(vec) #On initialise le vecteur des solutions pour tous les modes


x = np.linspace(0.,L_tot,N_elts+1); #Vecteur espace
fig,ax = plt.subplots();
for i in range(4): #On veut tracer les 4 premiers modes
    tracer_mode(x,ax,i,sol) #Tracé de la déformée du mode i+1

ax.grid(which='both')

#Pour les valeurs idéales
fig, axs = plt.subplots()
x = np.linspace(0.,15.,200)
y = f(x)
axs.set_xlabel('kL',fontsize=18)
axs.set_ylabel('Amplitude',fontsize=18)
axs.plot(x,y,label=r"$cos(kL)-1/cosh(kL)$");
axs.grid(which='both')
axs.legend(fontsize=18)

kl = dichotomie(f,13.0,15.0)
w = (kl**2)*np.sqrt(E*I/(rho*S*(L_tot**4)))
f = w/(2.*np.pi)
print("kl : ",kl)
print("omega = ",w)
print("f = ",f)
