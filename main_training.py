# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 18:08:39 2020

@author: Col.Moutarde
"""

import Convolution_Layer.Convolution_Layer as cnn
#import Convolution_Layer.Convolution_Layer_cpu as cnn #use 100% of cpu
#import Convolution_Layer.Convolution_Layer_gpu as cnn #just bad avrg x3 slower
from PIL import Image
import time
import random
import numpy as np
import concurrent.futures
from itertools import repeat
import os

def vectorisation(n):
    e = np.zeros((7, 1))
    e[n] = 1.0
    return e

def load_list_image(liste, path):
    """Charger une image contenue dans la liste avec son vecteur corespondant
        Inputs:
            -liste : liste des images plus categories (list, type=(tuples, type=(str, int), size=2))
            -path  : path de l'image (type=str)"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        liste_img = [val for val in executor.map(image_load, liste, repeat(path, len(liste)))]
    return liste_img

def image_load(args, path):
    """Charger une image contenue dans la liste avec son vecteur corespondant
        Inputs:
            -args : image plus categorie (tuples, type=(str, int), size=2)
            -path : path de l'image (type=str)"""
    vector = np.zeros((7, 1))
    vector[args[1]] = 1.0
    return [np.array(Image.open('{}{}'.format(path, args[0])))/255, vector]

def CNN_loader(name, path, nag=0, cm=0):
    """Charger un cnn sauvgarder (pas de verification d'existance)
        Inputs:
            -name : version du cnn a charger (str)
            -path : path deu cnn (str)
        Kwargs:
            -nag  : valeur du la constant nag (float)
            -cm   : valeur du la constant cm (float)
        Output:
            -cnn object"""
    NN_obj = cnn.Convolution_NN() #l'objet reseau neronal
    NN_obj.load('{}'.format(name), '{}'.format(path))
    print("[]: Neural Network loaded located: {}{}".format(path, name))
    
    #parametrage si les valeurs de nag et cm son non nul
    if nag != 0:
        NN_obj.nag_bool = True
        NN_obj.m_constant = nag
    elif cm != 0:
        NN_obj.cm_bool = True
        NN_obj.m_constant = cm
    #creation des matrice de moment des poid et biase
    if nag != 0 or cm != 0:
        NN_obj.momentum_Weight = []
        NN_obj.momentum_Biase  = []
        for Object in NN_obj.Layer:
            if Object.bool_weight:
                NN_obj.momentum_Weight.append([np.zeros(w.shape) for w in Object.Weight])
            if Object.bool_biase:
                NN_obj.momentum_Biase.append([np.zeros(b.shape) for b in Object.Biase])
        print("[]: Momentum of the Neural Network is created")
    return NN_obj

def CNN_creation():
   
    #parametre du reseau
    layer_cnn  = (3, 24)
    r          = (4,)
    s          = (2,) 
    layer_full = (384, 16)
    layer_last = (layer_full[-1], 7)

    #l'architecture:
    #"""
    NN_obj = cnn.Convolution_NN(nag = 0.7)
    
    for i in range(len(layer_cnn)-1):
        NN_obj.add(cnn.Convolution(inputs=layer_cnn[i], filters=(r[i], r[i]), output=layer_cnn[i+1], padding=True, strides=s[i]))
        NN_obj.add(cnn.ReLU())
        NN_obj.add(cnn.Pooling(pool=(8, 8), padding=True, type_='Average'))
    NN_obj.add(cnn.Vectorisation())
    NN_obj.add(cnn.FullConnected(layer=layer_full))
    NN_obj.add(cnn.FullConnected(layer=layer_last, activation='Softmax'))
    NN_obj.cost_fonction = cnn.Vectorial_Loss('cross_entropy')
    print('[]: Neural Network is created')
    return NN_obj


    

if __name__ == '__main__':
    list_category = ['badland', 'field', 'forest', 'lake', 'mountain', 'ocean', 'street']
    
    #parametre generaux 
    size_image  = 64 #taille en pixel de l'image (64, 256)
    bool_save   = True

    liste_cnn = [("NNet_large_4", 0),
                 ("NNet_large_8", 0),
                 ("NNet_large_16", 0),
                 ("NNet_large_24", 0),
                 ("NNet_medium_4", 0),
                 ("NNet_medium_8", 0),
                 ("NNet_medium_16", 0),
                 ("NNet_medium_24", 0),
                 ("NNet_medium_32", 0),
                 ("NNet_small_4", 0),
                 ("NNet_small_8", 0),
                 ("NNet_small_16", 0),
                 ("NNet_small_24", 0),
                 ("NNet_small_32", 0),] # liste des cnn a entrainer


    #option d'aprentisage
    n_iteration = 30 #nombre d'iteration total de la base de donne
    size_batch  = 50 #taille des batchs (partie)
    
    #hyperparametre d'aprentisage
    epsilon     = 1e-3
    regretion   = 0.99
    nag         = 0.85
    cm          = 0
    
    cost_texte  = ''
    
    
    #les path sont different (variant en fonction de la taille)
    if size_image == 256:
        path_train = './train/'
        path_texte_train = './ocean6_train.txt'
        
        path_val = './val/'
        path_texte_val = './ocean6_val.txt' 
    
    elif size_image == 64:
        path_train = './train_64/'
        path_texte_train = './ocean6_train.txt'
        
        path_val = './val_64/'
        path_texte_val = './ocean6_val.txt'
    
           
    #ouverture de fichier texte contenat la liste des images
    with open(path_texte_train, 'r') as f:
        liste_train = [(i.split(' ')[0], int(i.split(' ')[1])) for i in f.read().split('\n')]
    print('[]: training texte is loaded')
    
    with open(path_texte_val, 'r') as f:
        liste_val = [(i.split(' ')[0], int(i.split(' ')[1])) for i in f.read().split('\n')]
    print('[]: validation texte is loaded')
    
    #si les images sont de taille 64 on peux les chargé dans la memoire
    if size_image == 64:
        liste_image_train = load_list_image(liste_train, path_train)
        liste_image_val = load_list_image(liste_val, path_val)
        print("[]: image loaded in memory")
    
    
    for nn_name, nn_number in liste_cnn:
        print("----------------------------------------")
        if nn_name in os.listdir("./{}".format(size_image)):
            NNet = CNN_loader('{}'.format(nn_number), './{}/{}/'.format(size_image, nn_name), nag=nag, cm=cm)
            epsilon *= regretion**nn_number
    
        else:
            #on cree le dossier ou l'on sauvegadra le network
            NNet = CNN_creation()
            nn_number = -1 #on commence a -1 car au debut de l'aprentisage on ajoute 1
            if bool_save:
                os.mkdir("./{}/{}".format(size_image, nn_name))
                #on cree le fichier ou l'on sauvegadra les costs
                with open('./{}/{}/cost.txt'.format(size_image, nn_name), 'w') as f:
                    f.write('')
                NNet.save('0', path='./{}/{}/'.format(size_image, nn_name))
        
        for i in range(0, n_iteration):
            #on shuffle les image pour les batches
            nn_number += 1
            if size_image == 64:
                random.shuffle(liste_image_train)                
            else:
                random.shuffle(liste_train)
            
            for k in range(0, len(liste_train), size_batch):
                t = time.time()
                
                if size_image == 64:
                    #on fait evoluer le reseau neuronal sur les images chargées
                    NNet.gradient_set(liste_image_train[k:k+size_batch], epsilon)   
                else:
                    #on charge les images plus rapidement a l'aide d'un thread
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        images = [val for val in executor.map(image_load, liste_train[k:k+size_batch], repeat(path_train, size_batch))]
                    
                    #on fait evoluer le reseau neuronal sur les images chargées
                    NNet.gradient_set(images, epsilon)
                
                t = time.time() - t
                
                if k%2000 == 0:
                    print("[{}][{}/35000]: Time is {} // Cost: {}".format(nn_number, k, round(t, 5), NNet.cost_learning/size_batch))
                    
                if bool_save: #on garde en memoire les cost que si on les sauvgarde
                    cost_texte += '{}\t{}\t{}\n'.format(nn_number, k, NNet.cost_learning/size_batch)
                NNet.cost_learning = 0
            
            epsilon *= regretion
            
            #save l'iteration du network
            if bool_save:
                NNet.save('{}'.format(nn_number), path='./{}/{}/'.format(size_image, nn_name))
            
            #save la liste des couts
            if bool_save:
                with open('./{}/{}/cost.txt'.format(size_image, nn_name), 'a') as f:
                    f.write(cost_texte)
                cost_texte = ''
            """
            #on colcule le taux de reussit avec les differente categorie
            if size_image == 64:
                evaluation = NNet.accuracy_eval(liste_image_val, type_='unic')[0]
            else:
                evaluation = NNet.accuracy_eval(load_list_image(liste_val, path_val), type_='unic')[0]
                
            #on somme puis divide par le nombre de categorie car chaque categorie a le meme nombre d'image
            print("accuracy: {}".format(np.sum(np.array(evaluation))/len(list_category)))
            
            for cat, res in zip(list_category, evaluation):
                #on afiche la reusite par categorie
                print('[{}]: {}'.format(cat, res))
            """
