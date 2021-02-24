# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:44:48 2019

@author: Victor Baconnet - victor.baconnet@hotmail.com

All rights reserved 

Calcul des modes de vibrations d'une cavité acoustique

"""

import matplotlib.pyplot as plt
import numpy as np


def tables(Lx=10,Ly=20,Nx=3,Ny=5,e='triangle'):
    """ Description : Creation a partir du nombre de point et de la longueur du domaine la table de coord. globale des noeuds et la table de connexion pour un maillage triangulaire

        Donnees :   * Lx - Float : Longueur selon la direction x
                    * Ly - Float : Longueur selon la direction y
                    * Nx - Int   : Nbre point de discretisation selon x
                    * Ny - Int   : Nbre point de discretisation selon y

        Resultats : * Noeud : Table coord. gloable Noeuds
                    * Tbc : Table de connexion
    """
    nx = Nx - 1 # Nbre element sur x
    ny = Ny - 1 # Nbre element sur y


    lx = np.linspace(0,Lx,Nx)
    ly = np.linspace(0,Ly,Ny)
    
    Noeud = np.zeros((Nx*Ny,2))
    if e=='triangle':
        Tbc = np.zeros((2*nx*ny,3),dtype='int')
    elif e == 'carre':
        Tbc = np.zeros((nx*ny,4),dtype='int') 

    Ne = 0
    i=0
    j=0

    while j < Ny-1:     # On se deplace sur les points sur y
        i = 0
        while i< Nx:  # On se deplace sur les points sur x
            if e == 'triangle':
                if 0<i and (Ne+1)%2 == 0:
                    A1=j*Nx+i
                    xA1 = lx[i]
                    yA1 = ly[j]
                    A2=(j+1)*Nx+i
                    xA2 = lx[i]
                    yA2 = ly[j+1]
                    A3=(j+1)*Nx+i-1
                    xA3 = lx[i-1]
                    yA3 = ly[j+1]

                elif 0<=i<=Nx-2:
                    A1=j*Nx+i
                    xA1 = lx[i]
                    yA1 = ly[j]
                    A2=j*Nx+i+1
                    xA2 = lx[i+1]
                    yA2 = ly[j]
                    A3=(j+1)*Nx+i
                    xA3 = lx[i]
                    yA3 = ly[j+1]
                    i=i+1

                elif (i+1)%Nx == 0:
                    break;

            elif e == 'carre':
                if (i+1)%Nx == 0:
                    break;
                else:
                    A1=j*Nx+i
                    xA1 = lx[i]
                    yA1 = ly[j]
                    A2=j*Nx+i+1
                    xA2 = lx[i+1]
                    yA2 = ly[j]
                    A3=(j+1)*Nx+i+1
                    xA3 = lx[i+1]
                    yA3 = ly[j+1]
                    A4=(j+1)*Nx+i
                    xA4 = lx[i]
                    yA4 = ly[j+1]
                    i=i+1

            if e == 'triangle':
                Tbc[Ne,0]=int(A1)
                Tbc[Ne,1]=int(A2)
                Tbc[Ne,2]=int(A3)
            elif e == 'carre':
                Tbc[Ne,0]=int(A1)
                Tbc[Ne,1]=int(A2)
                Tbc[Ne,2]=int(A3)
                Tbc[Ne,3]=int(A4)

            Noeud[A1,0] = xA1
            Noeud[A1,1] = yA1
            Noeud[A2,0] = xA2
            Noeud[A2,1] = yA2
            Noeud[A3,0] = xA3
            Noeud[A3,1] = yA3

            if e == 'carre':
                Noeud[A4,0] = xA4
                Noeud[A4,1] = yA4
            Ne = Ne + 1 # Numero de element
        j=j+1
    return Tbc,Noeud


def dist(y1,y2,y3,y4):
    #Pour faciliter l'écriture dans l'initialisation des matrices élémentaires
    return (y1-y2)*(y3-y4)

def calculMatricesElement(c1,c2,c3,c): 
    """
    Calcule les matrices de masse et raideur pour un élément triangulaire dont
    les extrémités sont c1, c2, c3 (dans le sens trigo)
    
    Paramètres d'entrée:
        
        - c1,c2 et c3 sont des itérables (tuple, liste) qui contiennent les coordonnées
        des sommets de l'élément triangulaire considéré (coordonnées en x et en y).
        Exemple : c1 = [0.0, 0.0], c2 = [0.0, 0.5], c3 = [0.5, 0.0]
        
        - c : vitesse du son dans le fluide
        
    Retourne:
        - Kx : matrice de raideur x élémentaire (voir rapport) de taille (3,3)
        - Kz : matrice de raideur z élémentaire (voir rapport) de taille (3,3)
        - Me : matrice de masse élémentaire (voir rapport) de taille (3,3)
    
    """
    
    x1 = c1[0]; y1 = c1[1]
    x2 = c2[0]; y2 = c2[1]
    x3 = c3[0]; y3 = c3[1]
    
    a = np.array([[1.,x1,y1],
                  [1.,x2,y2],
                  [1.,x3,y3]])
    
    #++++++++ Initialisation des matrices Kx, Kz et M élémentaires +++++++++++++++
    Kx = np.zeros((3,3))
    Kz = np.zeros((3,3))
    Me = np.zeros((3,3))
    
    Kx[0,:] = np.array([dist(y2,y3,y2,y3),dist(y2,y3,y3,y1),dist(y2,y3,y1,y2)])
    Kx[1,:] = np.array([dist(y2,y3,y3,y1),dist(y3,y1,y3,y1),dist(y3,y1,y1,y2)])
    Kx[2,:] = np.array([dist(y2,y3,y1,y2),dist(y3,y1,y1,y2),dist(y1,y2,y1,y2)])
        
    Kz[0,:] = np.array([dist(x2,x3,x2,x3),dist(x2,x3,x3,x1),dist(x2,x3,x1,x2)])
    Kz[1,:] = np.array([dist(x2,x3,x3,x1),dist(x3,x1,x3,x1),dist(x3,x1,x1,x2)])
    Kz[2,:] = np.array([dist(x2,x3,x1,x2),dist(x3,x1,x1,x2),dist(x1,x2,x1,x2)])
    
    Me[0,:] = np.array([2,1,1])
    Me[1,:] = np.array([1,2,1])
    Me[2,:] = np.array([1,1,2])
    
    Kx = 1.0/(2.0 * np.linalg.det(a)) * Kx
    Kz = 1.0/(2.0 * np.linalg.det(a)) * Kz
    Me = np.linalg.det(a)/(c*24.0) * Me
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    return Kx,Kz,Me

def assembler2D(K, M, Ne, tbc, coord, c):
    """
    Assemble les matrices de raideur Kx,Kz et la matrice de masse Me élémentaires
    dans les matrices de raideur et de masse globales K et M.
    
    Paramètres d'entrée:
        - Ne : Nombre d'éléments
        - Npoints : Nombre de points
        - tbc : table de connexions
        - coord : table des coordonnées 
        - c : vitesse du son dans le fluide
        
    Retourne:
        - K : matrice de raideur globale de taille (Npoints,Npoints)
        - M : matrice de masse globale de taille (Npoints,Npoints)
    """

    # print(f"Nombre d'éléments : {Ne}")

    for e in range(Ne):
        print(e)
        Kx,Kz,Me = calculMatricesElement(coord[tbc[e,0]],coord[tbc[e,1]],coord[tbc[e,2]], c) #Génération des matrices élémentaires
        E = Kx + Kz #Regrouper les matrices élémentaires en une seule matrice pour simplifier
        F = Me
        for j in range(3):
            for k in range(3):
                K[tbc[e,j],tbc[e,k]] += E[j,k]
                M[tbc[e,j],tbc[e,k]] += F[j,k]
        
def remplirMatrice2D(n_x,n_y,P):
    """
    Transforme le vecteur solution obtenu par éléments finis en matrice 2D 
    pour les tracés.
    
    Paramètres d'entrée:
        - n_x : nombre de points en x
        - n_z : nombre de points en z
        - P   : vecteur solution 1D de longueur Npoints*Npoint
        
    Retourne:
        - Press2D : Matrice de taille (Npoints,Npoints)
    """
    Press2D = np.zeros((n_y,n_x))
    for i in range(n_y):
        Press2D[i,:] = P[i*n_x:i*n_x+n_x]
        
    return Press2D

def find_min(val):
    """
    Trouve la position du minimum d'un vecteur.
    
    Paramètres d'entrée:
        - val : vecteur 1D 
        
    Retourne:
        - index : la position du minimum dans le vecteur
        - mini  : le minimum du vecteur
        
    Exemple: Si val = [1,4,1.5,9], find_min(val) renvoie (2,1.5)
    """
    
    index = 0;
    mini = np.min(val);
    for i in val:
        if abs(mini-i) < 0.00000001:
            return index,mini
        else:
            index += 1
    return index, mini;

def trier_modes(val,vec):
    """
    Trie les modes dans l'ordre croissant car linalg.eig ne rend pas les 
    valeurs propres dans l'ordre
    
    Paramètres d'entrée:
        - val : vecteur des valeurs propres 
        - vec : matrice de vecteurs propres
    
    Retourne:
        - tri_val : les valeurs propres triées dans l'ordre croissant
        - tri_vec : les vecteurs propres associés aux valeurs propres triées
    """
    tri_val = np.copy(val); #Va contenir les valeurs propres triées
    tri_vec = np.copy(vec); #Va contenir les vecteurs propres triés, correspondant aux valeur propres
    i=0;
    while (len(val) != 0):
        index, mini = find_min(val); #On récupère le minimum des valeurs propres et l'index correspondant
        tri_val[i] = mini; 
        tri_vec[:,i] = vec[:,index];
        val = np.delete(val, index);
        vec = np.delete(vec, index, axis=1);
        i += 1

    return tri_val, tri_vec;


def tracer_pression_analytique(nx, nz, ax, fig, L, n_x, H, n_z):
    """
    Trace le champ de pression calculé avec la solution analytique sur une 
    figure de type subplots : fig,ax = plt.subplots()
    
    p(x,z) = cos(nx*PI*x/L) * cos(nz*PI*z/H)
        
    Paramètres d'entrée : 
        - nx  : mode de propagation en x
        - nz  : mode de propagation en z
        - ax  : axis du plt.subplots()
        - fig : fig du plt.subplots()
        - L   : dimension x de la cavité
        - n_x : nombre de points en x
        - H   : dimension z de la cavité
        - n_z : nombre de points en z
    """
    
    x , z = np.meshgrid(np.linspace(0.,L,n_x),np.linspace(0.,H,n_z))
    pression = np.cos(nx*np.pi*x/L)*np.cos(nz*np.pi*z/H);
    cset = ax.contourf(x,z,pression,100,cmap='jet')  
    fig.colorbar(cset, ax = ax)
    ax.set_xlabel('x (m)', fontsize=18)     
    ax.set_ylabel('z (m)', fontsize=18)
    ax.set_title(f'Mode ({nx},{nz}) analytique',fontsize=17)

def tracer_pression_numerique(pression, index_mode, ax, fig,  L, n_x, H, n_z, nx=None, nz=None ):
    """
    Trace le champ de pression calculé avec par éléments finis (vecteur propre) sur une 
    figure de type subplots : fig,ax = plt.subplots()
        
    Paramètres d'entrée : 
        - pression : la matrice des vecteurs propres
        - index_mode : l'index du mode à tracer (position dans la matrice des vecteurs propres)
        - nx  : mode de propagation en x
        - nz  : mode de propagation en z
        - ax  : axis du plt.subplots()
        - fig : fig du plt.subplots()
        - L   : dimension x de la cavité
        - n_x : nombre de points en x
        - H   : dimension z de la cavité
        - n_z : nombre de points en z
    """
    x,z = np.meshgrid(np.linspace(0.,L,n_x), np.linspace(0.,H,n_z))
    pression_a_tracer = remplirMatrice2D(n_x, n_z, pression[:,index_mode])
    cset = ax.contourf(x,z,pression_a_tracer, 100, cmap='jet')
    if nx is not None and nz is not None:
        ax.set_title(f'Mode ({nx},{nz}) EF',fontsize=18)
    fig.colorbar(cset, ax = ax)
    ax.set_xlabel('x (m)', fontsize=18)     
    ax.set_ylabel('z (m)', fontsize=18)
    
def tracer_comparaison(pression, index_mode, nx, nz, axs, fig, L, n_x, H, n_z):
    """
    Trace le champ de pression analytique et par éléments finis sur une figure de type
    plt.subplots avec 2 colonnes:
        
    fig, axs = plt.subplots(ncols = 2)
    
    Paramètres d'entrée : 
        - pression : la matrice des vecteurs propres
        - index_mode : l'index du mode à tracer (position dans la matrice des vecteurs propres)
        - ax  : axis du plt.subplots()
        - fig : fig du plt.subplots()
        - nx  : mode de propagation en x
        - nz  : mode de propagation en z
        - L   : dimension x de la cavité
        - n_x : nombre de points en x
        - H   : dimension z de la cavité
        - n_z : nombre de points en z
    
    """
    
    tracer_pression_analytique(nx, nz, axs[1], fig, L, n_x, H, n_z)
    tracer_pression_numerique(pression, index_mode, axs[0], fig, L, n_x, H, n_z, nx, nz)


#=================================================================================================================
 #                                        ASSEMBLAGE ET CALCULS
#=================================================================================================================

L = 1.0 #Dimension x
H = 0.5 #Dimension y
n_x = 3 #Nombre de points de discrétisation en x
n_z = 2 #Nombre de points de discrétisation en y
c = 340. #Vitesse du son dans l'air

[tbc,coord] = tables(Lx = L, Ly = H, Nx = n_x, Ny = n_z); #On récupère la table de coordonnées et table de connexion
Ne = len(tbc) #Nombre d'éléments
Npoints = len(coord) #Nombre de points

print(f"Calcul lancé avec {Ne} éléments")

#Assemblage de la matrice avec les éléments 2D

print("Assemblage des matrices K et M...")
K,M = np.zeros((Npoints,Npoints)),np.zeros((Npoints,Npoints))
assembler2D(K, M, Ne, tbc, coord, c)
print("Assemblage : fait")

print("Calcul des valeur propres...")
val, vec = np.linalg.eig(K.dot(np.linalg.inv(M))) #Valeurs propres (k^2) et vecteurs propres (pression)
print("Calcul des valeurs propres : fait")

#Enlever les valeurs négatives (du style -1e-14) qui valent en fait 0
for i in range(len(val)):
    if val[i] < 0.0:
        val[i]=0.


"""
Les valeurs propres correspondent aux valeurs de k^2. Les vecteurs propres sont 
donnés colonne par colonne. Exemple, mode1 : pression[:,0]

Les vecteurs propres et valeurs propres ne sont pas triées dans l'ordre. Pour les trier,
on utilise la fonction trier_modes() qui renvoie les valeurs propres triées avec les 
vecteurs propres correspondants.
"""

print("Tri des modes...")
omega_carre, pression = trier_modes(val,vec); #trier modes et valeurs propres dans l'ordre croissant
print("Tri des modes : fait")

f = np.sqrt(omega_carre)/(2.0*np.pi)

#==============================================================================
#                                   TRACE DES SOLUTIONS
#==============================================================================

#++++++ AJUSTER CE QU'ON VEUT TRACER ICI +++++++++++++
tracer_analytique = False #Mettre à True pour tracer les solutions analytiques
tracer_numerique = False #Mettre à True pour tracer les solutions numériques 
trace_comparaison = True #Mettre à True pour tracer une comparaison 

#----------------------------- TRACER LES SOLUTIONS ANALYTIQUES ------------------------
plt.close('all')

if (tracer_analytique):

    mode_x_max = 3 # Mode maximal en x à tracer
    mode_z_max = 3 # Mode maximal en z à tracer

    for nx in range(mode_x_max):
        for nz in range(mode_z_max):
            fig,ax = plt.subplots()
            tracer_pression_analytique(nx, nz, ax, fig, L, n_x, H, n_z)
            plt.pause(0.1)
            plt.tight_layout()
        
#---------------------------------------------------------------------------------------

#---------------------------------- TRACER SOLUTIONS NUMERIQUES ------------------------
if (tracer_numerique): 
    
    # Plage de modes à tracer
    index_mode_min = 1 
    index_mode_max = 16
    
    for index_mode in range(1,len(omega_carre[index_mode_min:index_mode_max])):    
        fig, ax = plt.subplots()
        tracer_pression_numerique(pression, index_mode, ax, fig, L, n_x, H, n_z)
        plt.pause(0.1)
        plt.tight_layout()
#---------------------------------------------------------------------------------------


#--------------------------------- TRACER COMPARAISON ----------------------------------

if (trace_comparaison):

    nx, nz = 2,2
    index_mode = 11
    
    fig, axs = plt.subplots(ncols=2, figsize=(13.0,7.0))
    tracer_comparaison(pression, index_mode, nx, nz, axs, fig, L, n_x, H, n_z)
    plt.pause(0.1)
    plt.tight_layout()

#---------------------------------------------------------------------------------------