# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:44:34 2019

@author: thibault
"""

import numpy as np
import concurrent.futures
from scipy import signal

class Base_Layer(object):  
    
    bool_weight = False #booleen servant a savoir si il y a dse poids dans ce layer
    bool_biase = False #booleen servant a savoir si il y a des biases dans ce layer
    
    def __init__(self, type_):
        """Base Object for layer"""
        self.type = type_ #variable pour la sauvgarde du modele
        self.kwargs = {}
    
    def __repr__(self):
        """repr"""
        if len(self.kwargs) == 0:
            return "{}()".format(self.type)
        
        return_txt = ""
        for key, arg in self.kwargs.items():
            return_txt += "{} : {}/t".format(key, arg)
            
        return "{}({})".format(self.type, return_txt[:-2])
    
    def __str__(self):
        """str"""
        return 'Layer : {}'.format(self.type)



class ReLU(Base_Layer):
    def __init__(self, **kwargs):
        """None lineare fontion Relu : x-->max(0, x)"""
        super(ReLU, self).__init__('ReLU')
        
        if kwargs:
            for key, value in kwargs.items():
                raise Exception("{} = {} not expected for kwargs".format(key, value))
    
    @staticmethod    
    def feedforward(X):
        """Calculation for the next layer 
            Inputs:
                -X : the vectorial image going to be convoluted (array)"""
        return np.maximum(0, X)
    
    @staticmethod
    def gradient_activation(Img_In, D_Img_Out):
        """Calculation for the parcial derivative of the previous parcial devivative layer
            Inputs:
                -Img_Ing  : the image who was convoluted (array)
                -D_Img_Out: the parcial devivation of the cost by the output of this convolution(array,shape=Ing_In.shape)"""
        assert Img_In.shape == D_Img_Out.shape, 'D_Img_Out shape expected : {} not {}'.format(Img_In.shape, D_Img_Out.shape)
        return 1.0*(Img_In>0)*D_Img_Out


class Pooling(Base_Layer):
    def __init__(self, **kwargs):
        """Pooling Layer.
            Kwarg required:
                -Pool_size : dimention of the filter (tuple, size=2, type=int)
            Kwargs:
                -padding : filling the missing intput by zeros (bool, defaut=False)
                -type_   : type of pooling : 'Max', 'Average', 'Sum' (str)"""
        super(Pooling, self).__init__('Pooling')
        
        if 'pool' in kwargs.keys():
            #la taille du pooling
            self.pool_size = kwargs.pop('pool')
            self.kwargs['pool'] = self.pool_size
            assert isinstance(self.pool_size, tuple), 'pool type expected tuple'
            assert len(self.pool_size) == 2, 'pool len expected is 2'
            assert isinstance(self.pool_size[0], int) and isinstance(self.pool_size[1], int), 'pool[0] and pool[1] type expected int'
        else:
            raise Exception("pool is required")
        
        if 'padding' in kwargs.keys():
            #padding permet de remplire les ligne et colone manquant par des zeros
            self.padding = kwargs.pop('padding')
            self.kwargs['padding'] = self.padding
            assert isinstance(self.padding, bool), 'padding type expected bool'
        else:
            #par defaut padding n'est pas activé
            self.padding = False
            self.kwargs['padding'] = False
        
        if 'type_' in kwargs.keys():
            #strides permet de 'saut n' case entre chaque filtre
            self.type_ = kwargs.pop('type_')
            self.kwargs['type_'] = self.type_
            assert self.type_ in ('Max', 'Average', 'Sum'), 'type_ not expected : {}'.format(self.type)
        else:
            #par defaut le saut est incrementiel de 1
            self.type_ = 'Max'
            self.kwargs['type_'] = 'Max'
        
        if self.type_ == 'Max':
            self.feedforward = self.feedforward_max
            self.gradient_activation = self.gradient_activation_max
            
            
        elif self.type_ == 'Average':
            self.feedforward = self.feedforward_fast
            self.gradient_activation = self.gradient_activation_fast
            
        elif self.type_ == 'Sum':
            self.feedforward = self.feedforward_fast
            self.gradient_activation = self.gradient_activation_fast
        
        if kwargs:
            for key, value in kwargs.items():
                raise Exception("{} = {} not expected for kwargs".format(key, value))
    
    def feedforward_fast(self, X):
        """Calculation for the next layer (only 'Average' and 'Sum' methodes)
            Inputs:
                -X : the image going tobe pool (array)"""

        if self.padding:
            shape_out = (max(1, (X.shape[0]-1)//self.pool_size[0]+1), max(1, (X.shape[1]-1)//self.pool_size[1]+1), X.shape[2])
            image = np.zeros((shape_out[0]*self.pool_size[0], shape_out[1]*self.pool_size[1], X.shape[2]))
            image[:X.shape[0], :X.shape[1], :] = X
            X = image
        else:
            shape_out = (X.shape[0]//self.pool_size[0], X.shape[1]//self.pool_size[1], X.shape[2])
        
        output_X = np.zeros(shape_out)
        
        for i in range(self.pool_size[0]):
            for j in range(self.pool_size[1]):
                output_X += X[i::self.pool_size[0], j::self.pool_size[1], :][:shape_out[0], :shape_out[1], :]
        
        if self.type_ == 'Average':
            return output_X/(self.pool_size[0]*self.pool_size[1])
        return output_X        
    
    def gradient_activation_fast(self, Img_In, D_Img_Out):
        """Calculation for the previous parcial devivative layer (before this convolution)(only 'Average' and 'Sum' methodes)
            Inputs:
                -Img_Ing  : the image who was convoluted (array, shape[0]=input_size)
                -D_Img_Out: the parcial devivation of the cost by the output of this convolution(array)"""
        delta_act = np.zeros(Img_In.shape)
        
        for i in range(self.pool_size[0]):
            for j in range(self.pool_size[1]):
                delta_act[i::self.pool_size[0], j::self.pool_size[1], :] = D_Img_Out[:(Img_In.shape[0]-i-1)//self.pool_size[0]+1, :(Img_In.shape[1]-j-1)//self.pool_size[1]+1, :]
                
        
        if self.type_ == 'Average':
            return delta_act/(self.pool_size[0]*self.pool_size[1])
        return delta_act
    
    def feedforward_max(self, X):
        """Calculation for the next layer (only 'Max' methode)
            Inputs:
                -X : the image going tobe pool (array)"""
                
        if self.padding:
            shape_out = (max(1, (X.shape[0]-1)//self.pool_size[0]+1), max(1, (X.shape[1]-1)//self.pool_size[1]+1), X.shape[2])
        else:
            shape_out = (X.shape[0]//self.pool_size[0], X.shape[1]//self.pool_size[1], X.shape[2])
        
        output_X = np.zeros(shape_out)
        
        
        for i in range(shape_out[0]):
            for j in range(shape_out[1]):
                for k in range(shape_out[2]):
                    output_X[i, j, k] = np.max(X[self.pool_size[0]*i:self.pool_size[0]*(i+1)-1, self.pool_size[1]*j:self.pool_size[1]*(j+1)-1, k])
        return output_X
    
    def gradient_activation_max(self, Img_In, D_Img_Out):
        """Calculation for the previous parcial devivative layer (before this convolution) (only 'Max' methode)
            Inputs:
                -Img_Ing  : the image who was convoluted (array, shape[0]=input_size)
                -D_Img_Out: the parcial devivation of the cost by the output of this convolution(array)"""
        delta_act = np.zeros(Img_In.shape)
        
        for i in range(D_Img_Out.shape[0]):
            for j in range(D_Img_Out.shape[1]):
                for k in range(D_Img_Out.shape[2]):
                    delta_act[self.pool_size[0]*i:self.pool_size[0]*(i+1)-1, self.pool_size[1]*j:self.pool_size[1]*(j+1)-1, k] = D_Img_Out[i, j, k]*self.max_prime(Img_In[self.pool_size[0]*i:self.pool_size[0]*(i+1)-1, self.pool_size[1]*j:self.pool_size[1]*(j+1)-1, k])
        return delta_act
    
    @staticmethod
    def max_prime(X):
        s = np.zeros(X.shape)
        s[np.unravel_index(np.argmax(X), X.shape)] = 1
        return s

class Convolution(Base_Layer):
    
    bool_weight = True #booleen servant a savoir si il y a dse poids dans ce layer
    
    def __init__(self, **kwargs):
        """Convolution Layer. 
            Kwarg required: 
                -Intput  : dimension of the input layer (int)
                -Filter  : size of the filter (tuple, size=2, type=int)
                -Output  : dimension of the output layer (int)
            Kwargs:
                -padding : filling the missing intput by zeros (bool, defaut=False)
                -strides : number of pixels shifts over the input matrix (int)"""
        
        super(Convolution, self).__init__('Convolution')
        #parametre de l'entrée et des sorties

        #creation du filtre
        if 'inputs' in kwargs.keys():
            #la taille de l'entré
            self.input_size = kwargs.pop('inputs')
            self.kwargs['inputs'] = self.input_size
            assert isinstance(self.input_size, int), 'inputs type expected int'
        else:
            raise Exception("inputs is required")
        
        if 'filters' in kwargs.keys():
            #la taille de des filtres
            self.filter_size = kwargs.pop('filters')
            self.kwargs['filters'] = self.filter_size
            assert isinstance(self.filter_size, tuple), 'filter type expected tuple'
            assert len(self.filter_size) == 2, 'filter len expected is 2'
            assert isinstance(self.filter_size[0], int) and isinstance(self.filter_size[1], int), 'filter[0] and filter[1] type expected int'
        else:
            raise Exception("filters is required")
        
        if 'output' in kwargs.keys():
            #la taille de la sortie
            self.output_size = kwargs.pop('output')
            self.kwargs['output'] = self.output_size
            assert isinstance(self.output_size, int), 'output type expected int'
        else:
            raise Exception("output is required")
        
        self.Weight = [np.random.standard_normal((self.filter_size[0], self.filter_size[1], self.input_size))*np.sqrt(2.0/(self.filter_size[0]*self.filter_size[1])) for i in range(self.output_size)]
        
        if 'strides' in kwargs.keys():
            #strides permet de 'saut n' case entre chaque filtre
            self.strides = kwargs.pop('strides')
            self.kwargs['strides'] = self.strides
            assert isinstance(self.strides, int), 'strides type expected int'
        else:
            #par defaut le saut est incrementiel de 1
            self.strides = 1
            self.kwargs['strides'] = 1
        
        if 'padding' in kwargs.keys():
            #padding permet de remplire les ligne et colone manquant par des zeros
            self.padding = kwargs.pop('padding')
            self.kwargs['padding'] = self.padding
            assert isinstance(self.padding, bool), 'padding type expected bool'
        else:
            #par defaut padding n'est pas activé
            self.padding = False
            self.kwargs['padding'] = False
        
        
        if kwargs:
            for key, value in kwargs.items():
                raise Exception("{} = {} not expected for kwargs".format(key, value))
    
    def size_output(self, Size_Input):
        """Size of output of the fonction feedforward 
            Inputs:
                -Size_Input : the size of the input of feedforward (tuple, size=3, type=int)"""
        assert Size_Input[2] == self.input_size, 'Size_Input[2] dimention expected : {} not {}'.format(self.input_size, Size_Input[2])
        if self.padding:
            return (max(1, (Size_Input[0]-self.filter_size[0]-1)//self.strides+2), max(1, (Size_Input[1]-self.filter_size[1]-1)//self.strides+2), self.output_size)
        
        #on verifie de l'imga de sortie peux etre crée
        assert Size_Input[0] >= self.filter_size[0] and Size_Input[1] >= self.filter_size[1], '(Size_Input[0], Size_Input[1]) size expected min : {}'.format(self.filter_size)
        return ((Size_Input[0]-self.filter_size[0])//self.strides+1, (Size_Input[1]-self.filter_size[1])//self.strides+1, self.output_size)
    
    
    def feedforward(self, X):
        """Calculation for the next layer (after this convolution)
            Inputs:
                -X : the image going to be convoluted (array, shape[0]=input_size)"""
        
        if self.strides == 1:
            output_X = np.zeros(self.size_output(X.shape))
            
            for k in range(len(self.Weight)):
                for n in range(self.Weight[k].shape[2]):
                    output_X[:, :, k] += signal.convolve2d(X[:, :, n], self.Weight[k][:, :, n], mode='valid')
            
        else:
            if self.padding:
                shape_out = self.size_output(X.shape)
                image_add = ((self.filter_size[0]-X.shape[0])%self.strides, (self.filter_size[1]-X.shape[1])%self.strides)
                image = np.zeros((X.shape[0]+image_add[0], X.shape[1]+image_add[1], self.input_size))
                image[:X.shape[0], :X.shape[1], :] = X
                X = image
            else:
                shape_out = self.size_output(X.shape)
            
            output_X = np.zeros(shape_out)
            
            #"""
            for k in range(len(self.Weight)):
                for i in range(self.Weight[k].shape[0]):
                    for j in range(self.Weight[k].shape[1]):
                        output_X[:, :, k] += np.sum(X[i:X.shape[0]+i-self.Weight[k].shape[0]+1:self.strides, j:X.shape[1]+j-self.Weight[k].shape[1]+1:self.strides, :]*self.Weight[k][i, j, :], axis=2)
        
        """
        for i in range(shape_out[0]):
            for j in range(shape_out[1]):
                for k in range(shape_out[2]):
                    output_X[i, j, k] = np.sum(X[self.strides*i:self.strides*i+self.filter_size[0], self.strides*j:self.strides*j+self.filter_size[1], :]*self.Weight[k])
        #"""
        return output_X


    def gradient_weight(self, Img_In, D_Img_Out):
        """Calculation for the parcial derivative of the Weight of this layer
            Inputs:
                -Img_Ing  : the image who was convoluted (array, shape[0]=input_size)
                -D_Img_Out: the parcial devivation of the cost by the output of this convolution(array, shape=output_size**)
                
            **; can be find with the following code: self.size_output(Img_In.shape)"""
        assert self.size_output(Img_In.shape) == D_Img_Out.shape, 'D_Img_Out shape expected : {} not {}'.format(self.size_output(Img_In.shape), D_Img_Out.shape)
        
        delta_Weight = [np.zeros(self.Weight[0].shape) for i in range(self.output_size)]

        for s in range(self.output_size):
            for k in range(self.Weight[0].shape[0]):
                for l in range(self.Weight[0].shape[1]):
                    for n in range(self.Weight[0].shape[2]):
                        if self.padding:
                            I = Img_In[k::self.strides, l::self.strides, n][:D_Img_Out.shape[0], :D_Img_Out.shape[1]]
                            delta_Weight[s][k, l, n]=np.sum(I*D_Img_Out[:I.shape[0], :I.shape[1], s])
                        else:
                            D = D_Img_Out[s]
                            delta_Weight[s][k, l, n]=np.sum(Img_In[k::self.strides, l::self.strides, n][:D.shape[0], :D.shape[1], n]*D)
        return delta_Weight

    def gradient_activation(self, Img_In, D_Img_Out):        
        """Calculation of the previous parcial devivative layer (before this convolution)
            Inputs:
                -Img_Ing  : the image who was convoluted (array, shape[0]=input_size)
                -D_Img_Out: the parcial devivation of the cost by the output of this convolution(array, shape=output_size**)
                
            **; can be find with the following code: self.size_output(Img_In.shape)"""
        assert self.size_output(Img_In.shape) == D_Img_Out.shape, 'D_Img_Out shape expected : {} not {}'.format(self.size_output(Img_In.shape), D_Img_Out.shape)
        
        delta_act  = np.zeros(Img_In.shape)
        
        for i in range(self.filter_size[0]):
            for j in range(self.filter_size[1]):
                for k in range(self.input_size):
                    if self.padding:
                        delta_act[i:i+self.strides*D_Img_Out.shape[0]:self.strides, j:j+self.strides*D_Img_Out.shape[1]:self.strides, k]  += np.sum(np.array([self.Weight[n][i,j,k]*D_Img_Out[:D_Img_Out.shape[0]-i%self.strides, :D_Img_Out.shape[1]-j%self.strides, n] for n in range(self.output_size)]), axis=0)
                    else:
                        delta_act[i:i+self.strides*D_Img_Out.shape[0]:self.strides, j:j+self.strides*D_Img_Out.shape[1]:self.strides, k]  += np.sum(np.array([self.Weight[n][i,j,k]*D_Img_Out[:, :, n] for n in range(self.output_size)]), axis=0)
        return delta_act

class Vectorisation(Base_Layer):
    def __init__(self, **kwargs):
        """Transforma 3d space inforamtion in to a 1d vecteur"""
        super(Vectorisation, self).__init__('Vectorisation')
        
        if kwargs:
            for key, value in kwargs.items():
                raise Exception("{} = {} not expected for kwargs".format(key, value))
    
    @staticmethod
    def feedforward(X):
        """Calculation for the next layer 
            Inputs:
                -X : the image going to be convoluted (array)"""
        assert len(X.shape) == 3, 'Image len shape expected : 3 not {}'.format(len(X.shape))
        return X.reshape((X.shape[0]*X.shape[1]*X.shape[2], 1))
    
    @staticmethod
    def gradient_activation(Img_In, D_Img_Out):
        """Calculation for the parcial derivative of the previous parcial devivative layer
            Inputs:
                -Img_Ing  : the image who was convoluted (array)
                -D_Img_Out: the parcial devivation of the cost by the output of this convolution(array, shape=Ing_In.shape)"""
        return D_Img_Out.reshape(Img_In.shape)

class FullConnected(Base_Layer):
    
    bool_weight = True #booleen servant a savoir si il y a dse poids dans ce layer
    bool_biase  = True #booleen servant a savoir si il y a des biases dans ce layer
    
    def __init__(self, **kwargs):
        """definition de l'object pour un reseau de neuron multiple.
            Kwarg required:
                -layer_size   : forme des layers de neurones (list, type=int)
            Kwargs:
                -activation   : fonction non-linearie a utiliser. Possible: ['sigmoid', 'relu'] (str, defaut='relu')"""
        super(FullConnected, self).__init__('FullConnected')
        
        if 'layer' in kwargs.keys():
            #la forme des layers de neurones
            layer_size = kwargs.pop('layer')
            self.kwargs['layer'] = layer_size
            assert isinstance(layer_size, tuple), 'layer type expected tuple'
        else:
            raise Exception("layer is required") 
        
        self.weight_size = [(layer_size[i+1], layer_size[i]) for i in range(len(layer_size)-1)]
        
        #on inicialisie les poids suivant une loi normale *sqrt(2/n) et les poids nul
        self.Weight = [np.random.standard_normal((l, c)) * np.sqrt(2.0/c) for l, c in self.weight_size]
        self.Biase  = [np.zeros((l, 1)) for l in layer_size[1:]]
        
        #on traite l'option activation    
        if 'activation' in kwargs.keys():
            activation = kwargs.pop('activation').lower()
            self.kwargs['activation'] = activation
            if activation == 'sigmoid':
                #fontion sigmoid : x --> 1/(1+exp(x))
                self.activation       = self.sigmoid
                self.activation_prime = self.sigmoid_prime
            elif activation == 'relu':
                #fontion ReLU : x --> max(0, x)
                self.activation       = self.ReLU
                self.activation_prime = self.ReLU_prime
            elif activation == 'softmax':
                #fonction softmax : x -> exp(x)/sum(exp(x))
                self.activation       = self.softmax
                self.activation_prime = self.softmax_prime
            else:
                raise Exception("{} not expected for activation".format(activation))
        else:
            #par defaut la fontion ReLU est utilisé
            self.activation       = self.ReLU
            self.activation_prime = self.ReLU_prime
            self.kwargs['activation'] = 'relu'
        
        if kwargs:
            for key, value in kwargs.items():
                raise Exception("{} = {} not expected for kwargs".format(key, value))
    
    def feedforward(self, X):
        """Calculation for the next layer 
            Inputs:
                -X : the image going to be convoluted (array)"""
        for w, b in zip(self.Weight, self.Biase):
            X = self.activation(np.dot(w, X) + b)
        return X
    
    def gradient_weight(self, Img_In, D_Img_Out):
        """Calculation for the parcial derivative of the weight 
            Inputs:
                -Img_Ing  : the image who was convoluted (array)
                -D_Img_Out: the parcial devivation of the cost by the output of this convolution(array, shape=Ing_In.shape)"""
        
        delta_Biase  = [np.zeros(b.shape) for b in self.Biase]
        delta_Weight = [np.zeros(w.shape) for w in self.Weight]
        
        activations = [Img_In]  #liste des vecteurs activation
        Zs          = []        #liste des vecteur avant la fontion non lineaire
        H           = Img_In    #le vecteur actuelement utile pour calculé le reste
        
        for w, b in zip(self.Weight, self.Biase):     #calcule de tous les vecteurs activation
            H = np.dot(w, H) + b
            Zs.append(H)
            H = self.activation(H)
            activations.append(H)
        
        grad_C  = D_Img_Out
        Z_prime = self.activation_prime(Zs[-1])
        delta   = grad_C * Z_prime
        delta_Biase[-1]  = delta
        delta_Weight[-1] = np.dot(delta, np.transpose(activations[-2]))
        
        for i in range(2, len(self.Weight)+1):
            Z_prime = self.activation_prime(Zs[-i])
            delta   = np.dot(np.transpose(self.Weight[-i+1]), delta) * Z_prime
            delta_Biase[-i]  = delta
            delta_Weight[-i] = np.dot(delta, np.transpose(activations[-i-1]))

        return (delta_Weight, delta_Biase)
    
    def gradient_activation(self, Img_In, D_Img_Out):
        """Calculation for the parcial derivative of the previous layer
            Inputs:
                -Img_Ing  : the image who was convoluted (array)
                -D_Img_Out: the parcial devivation of the cost by the output of this convolution(array, shape=Ing_In.shape)"""

        H = Img_In    #le vecteur actuelement utile pour calculé le reste
        Zs          = [H]        #liste des vecteur avant la fontion non lineaire
        
        for w, b in zip(self.Weight, self.Biase):     #calcule de tous les vecteurs activation
            H = np.dot(w, H) + b
            Zs.append(H)
            H = self.activation(H)
        
        Z_prime = self.activation_prime(Zs[-1])
        delta   = D_Img_Out * Z_prime
        
        for i in range(len(self.Weight)-1, -1, -1):
            Z_prime = self.activation_prime(Zs[i])
            delta   = np.dot(np.transpose(self.Weight[i]), delta) * Z_prime

        return delta
    
    @staticmethod
    def sigmoid(X):
        return 1/(1 + np.exp(-X))
    
    @staticmethod
    def sigmoid_prime(X):
        Y = 1/(1 + np.exp(-X))
        return Y*(1-Y)
    
    @staticmethod
    def softmax(X):
        exps = np.exp(X - np.max(X))
        return exps/np.sum(exps)
    
    @staticmethod
    def softmax_prime(X):
        exps = np.exp(X-np.max(X))
        p = exps/np.sum(exps)
        return p*(1-p)
    
    @staticmethod
    def ReLU(X):
        return np.maximum(0, X)
    
    @staticmethod
    def ReLU_prime(X):
        return 1.0*(X>=0)

class Vectorial_Loss:
    def __init__(self, Fonction):
        """fonction cout utilisé
            Input:
                -Fonction : fonction utilisé. Possible: ['l2', 'cvm', cvm_class', 'softmax'] (str)"""
        self.type = Fonction
        
        if Fonction == 'l2':
            #la norme L2 au carré
            self.feedforward = self.cost_L2    
            self.gradient    = self.cost_L2_grad
        elif Fonction == 'cvm':
            #la fonction CVM pour une classification multiple
            self.feedforward = self.cost_CVM    
            self.gradient    = self.cost_CVM_grad
        elif Fonction == 'cvm_class':
            #la fonction CVM pour une classification
            self.cost_delta = 1.0
            self.feedforward = self.cost_CVM_class    
            self.gradient    = self.cost_CVM_class_grad
        elif Fonction == 'softmax':
            #la fontion logaritmique de la normalisation expodentielle
            self.feedforward = self.cost_Softmax    
            self.gradient    = self.cost_Softmax_grad
        elif Fonction == 'cross_entropy':
            #la fontion cross entropy (softmax généralisé)
            self.feedforward = self.cost_Cross_Entropy    
            self.gradient    = self.cost_Cross_Entropy_grad
        else:
            raise Exception("{} not expected for loss".format(Fonction))
    
    @staticmethod
    def cost_L2(X, Y_expected):
        """fonction cout a l'aide de la norme L2²
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        #c'est une fontion utilisable uniquement dans des cas de regretion
        return np.sum((X-Y_expected)**2)
     
    @staticmethod
    def cost_L2_grad(X, Y_expected):
        """Gradien de la fonction cout a l'aide de la norme L2²
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        #c'est une fontion utilisable uniquement dans des cas de regretion
        return X - Y_expected
    
    def cost_CVM_class(self, X, Y_expected):
        """fontion CVM classification
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        #c'est une fonction de classification : chaque entree doit appartentir a une unique classe
        return np.sum(np.maximum(0, X*(1-Y_expected) +  (self.cost_delta - X[np.where(Y_expected == 1)][0])*(1-Y_expected)))
    
    def cost_CVM_class_grad(self, X, Y_expected):
        """Gradien de la fontion CVM classification
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        #c'est une fonction de classification : chaque entree doit appartentir a une unique classe
        #c'est a dire qu'il ne peux y avoir une seul valeur positif pour Y_expected
        j = X[np.where(Y_expected == 1)][0]
        Y = (X+(self.cost_delta-j)*(1-Y_expected)>0)
        return 1.0*Y  + np.sum(-1.0*Y)*Y_expected
    
    @staticmethod
    def cost_CVM(X, Y_expected):
        """Gradien de la fontion CVM
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        return np.sum(np.maximum(0, 1.0 - Y_expected*X))
    
    @staticmethod
    def cost_CVM_grad(X, Y_expected):
        """Gradien de la fontion CVM"""
        return 1.0*(1-X*Y_expected>0)
    
    @staticmethod
    def cost_Cross_Entropy(X, Y_expected):
        """la fontion 'cross entropy' generalisation de softmax
            /!/ la probabilisation n'est pas fait au préalable  
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        return -np.sum(Y_expected*np.log(X)+(1-Y_expected)*np.log(1-X))
    
    @staticmethod
    def cost_Cross_Entropy_grad(X, Y_expected):
        """Gradien de la fontion 'cross entropy' generalisation de softmax
            /!/ la probabilisation n'est pas fait au préalable  
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        return -Y_expected/X + (1-Y_expected)/(1-X)
        
    @staticmethod
    def cost_Softmax(X, Y_expected):
        """la fontion 'cross entropy (softmax)' pour une unique valeur egals a 1 and Y expected
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        #c'est une fonction de classification : chaque entree doit appartentir a une unique classe
        exps = np.exp(X - np.max(X))
        return -np.sum(Y_expected*(np.log(exps)-np.log(np.sum(exps))))
    
    @staticmethod
    def cost_Softmax_grad(X, Y_expected):
        """Gradien de la fontion 'cross entropy (softmax)' pour une unique valeur egals a 1 and Y expected
            Inputs:
                -Y          : valeur de sortie du mlp (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du mlp (narray, size=(1, .), type=float)"""
        #c'est une fonction de classification : chaque entree doit appartentir a une unique classe
        #c'est a dire qu'il ne peux y avoir une seul valeur positif pour Y_expected
        #On prend la precotion de ne pas depasser la valeur maxima des floot en soustraillent la valeur max de Y
        exps = np.exp(X - np.max(X))
        return exps/np.sum(exps) - Y_expected


class Convolution_NN:
    def __init__(self, **kwargs):
        """definition de l'object pour un reseau de neuron multiple.
            Kwargs:
                -loss         : fonction cost a utiliser (object, defaut=None)
                -cm           : utilisation du moment de base (float, entre [0, 1])
                -nag          : utilisation du moment de nesterov (float, entre [0, 1])"""
        self.Layer = []
        
        #on calcule le cost pendant le learning pour gagnier du temps
        self.cost_learning = 0
        #on traite l'option cost
        if 'cost' in kwargs.keys():
            self.cost_fonction = kwargs.pop('cost')
        else:
            #par defaut il n'y a pas de fonction
            self.cost_fonction = None
        
        #option du moment classique(cm) 
        if 'cm' in kwargs.keys():
            self.cm_bool = True
            constant = kwargs.pop('cm')
            assert 0 <= constant <= 1, 'cm expected between 0 and 1'
            self.m_constant = constant
            self.momentum_Weight = []
            self.momentum_Biase  = []
        else :
            self.cm_bool = False
        
        #option de l'aceleration du moment de nesterov
        if 'nag' in kwargs.keys():
            self.nag_bool = True
            constant = kwargs.pop('nag')
            assert 0 <= constant <= 1, 'cm expected between 0 and 1'
            self.m_constant = constant
            self.momentum_Weight = []
            self.momentum_Biase  = []
        else :
            self.nag_bool = False
        
        if kwargs:
            for key, value in kwargs.items():
                raise Exception("{} = {} not expected for kwargs".format(key, value))
    
    def __len__(self):
        """on retourne avec len la taille du reseau neuronal"""
        return len(self.Layer)
                
    def add(self, Object):
        """fonction permetant d'ajoute un nouveau layer
            Input:
                -Object : nouveau layer (object)"""
        self.Layer.append(Object)
        
        if self.cm_bool or self.nag_bool:
            if Object.bool_weight:
                self.momentum_Weight.append([np.zeros(w.shape) for w in Object.Weight])
            if Object.bool_biase:
                self.momentum_Biase.append([np.zeros(b.shape) for b in Object.Biase])
               
    def save(self, Name, **kwargs):
        """fonction permetant de sauvegardé le reseau de neurone
            Input:
                -Name : nom du reseau neuronal
            Kwarg:
                -path : chemin de sauvegarde"""
        
        if 'path' in kwargs.keys():
            #option d'un chemin de sauvgarde
            path = kwargs.pop('path')
        else:
            path = ''
            
        text = '{}'.format(self.cost_fonction.type)
        
        for obj in self.Layer:
            text += '\n{}\t{}'.format(obj.type, obj.kwargs)
        
        #sauvegarde du texte des option du reseau neuronal
        fichier = open('{}{}.txt'.format(path, Name), 'w+')
        fichier.write(text)
        fichier.close()
        
        #sauvgarde des poids et biases des layers
        #format path/Name_i_(Weight/Biase)
        for i in range(len(self.Layer)):
            if self.Layer[i].bool_weight:
                np.savez('{}{}_{}_Weight.npz'.format(path, Name, i), *self.Layer[i].Weight)
            if self.Layer[i].bool_biase:
                np.savez('{}{}_{}_Biase.npz'.format(path, Name, i), *self.Layer[i].Biase)
    
    def load(self, Name, Path):
        """"""
        fichier = open('{}{}.txt'.format(Path, Name), 'r')
        text = fichier.read()
        fichier.close()
        
        text = text.split('\n')
        self.cost_fonction = Vectorial_Loss(text[0]) # la premiere ligne les la pour seulment 
        
        self.Layer = [] #on reset les layers
        
        #on load l'architecture du reseau de neurone
        for t in text[1:]:
            i = t.split('\t')
            #on ajoute un nouveau layer par ligne
            if i[0] == 'ReLU':
                self.Layer.append(ReLU(**eval(i[1])))
            elif i[0] == 'Pooling':
                self.Layer.append(Pooling(**eval(i[1])))
            elif i[0] == 'Convolution':
                self.Layer.append(Convolution(**eval(i[1])))
            elif i[0] == 'Vectorisation':
                self.Layer.append(Vectorisation(**eval(i[1])))
            elif i[0] == 'FullConnected':
                self.Layer.append(FullConnected(**eval(i[1])))
        
        #on load les valeur des differents poids et biases
        for i in range(len(self.Layer)):
            if self.Layer[i].bool_weight:
                w = np.load('{}{}_{}_Weight.npz'.format(Path, Name, i))
                keys = w.files
                for k in range(len(self.Layer[i].Weight)):
                    self.Layer[i].Weight[k] = np.copy(w[keys[k]])
                w.close()
            if self.Layer[i].bool_biase:
                b = np.load('{}{}_{}_Biase.npz'.format(Path, Name, i))
                keys = b.files
                for k in range(len(self.Layer[i].Biase)):
                    self.Layer[i].Biase[k] = np.copy(b[keys[k]])
                b.close()  
    
    def feedforward(self, X):
        """Fonction du reseau de neurone
            Input:
                -X : valeur d'entré à calculé (narray, size=(1, .), type=float)"""
        for obj in self.Layer:
            X = obj.feedforward(X)
        return X
    
    def cost_set(self, Data):
        """fonction cout d'un set de data
            Input:
                -Data : liste de data a calculé le gradient (list, type=(list, type=narray, size=2))"""
        Cn = [self.cost_fonction.feedforward(self.feedforward(X), Y) for X, Y in Data]
        return sum(Cn)/len(Data)
    
    def accuracy_eval(self, Data, **kwargs):
        """permet d'evaluer la justesse de du mlp
            Input:
                -Data : data pour les tests (list, type=(list, type=narray, size=2))
            Kwargs:
                -mode : soit on prend la valeur max ('max') ou les valeur superieur a 0.5 ('sub')
                -type_ : soit on veut toute les valeurs bonne ('all') ou on considère chaque option ('unic')"""
        
        if 'mode' in kwargs.keys():
            mode = kwargs.pop('mode')
            assert mode in ('max', 'sub'), 'mode expected : sud or max'
        else:
            mode = 'max'
        
        if 'type_' in kwargs.keys():
            type_ = kwargs.pop('type_')
            assert type_ in ('all', 'unic'), 'type_ expected : all or unic'
        else:
            type_ = 'unic'
        
        if kwargs:
            for key, value in kwargs.items():
                raise Exception("{} = {} not expected for kwargs".format(key, value))
          
        len_ = 1
        for i in Data[0][1].shape:
            len_ *= i
        
        if  type_ == 'unic':
            output      = [0]*len_
            compteur    = [0]*len_
        else:
            output      = 0
            compteur    = len(Data)
        
        for index in Data:        
            y, y_exp = self.feedforward(index[0]), index[1]
            if type_ == 'unic':
                for i in np.where(y_exp == 1)[0]:
                    compteur[i] +=1
            
            if mode == 'max':
                if type_ == 'all':
                    if np.argmax(y) == np.where(y_exp == 1)[0][0]:
                        output += 1
                if type_ == 'unic':
                    if np.argmax(y) == np.where(y_exp == 1)[0][0]:
                        output[np.where(y_exp == 1)[0][0]] += 1
            elif mode == 'sub':
                if type_ == 'all':
                    if np.array_equal(np.where(y>0.5)[0], np.where(y_exp == 1)[0]):
                        output += 1
                elif type_ == 'unic':
                    for i in np.where(y_exp == 1)[0]:
                        if y[i][0] > 0.5:
                            output[i] +=1
        
        if type_ == 'unic':               
            for j in range(len(compteur)):
                if compteur[j] != 0:    
                    output[j] /= compteur[j]
            return output, compteur
        
        return output/compteur, compteur
    
    def gradient_set(self, Data, Epsilon):
        """fonction permetant de faire evoluer le reseau de neurons
            Inputs:
                -Data    : data pour l'apprentisage (list, type=(list, type=narray, size=2))
                -Epsilon : hyper-parametre, constant d'apprentisage (float)"""
        
        nabla_Weight = []
        for obj in self.Layer:
            if obj.bool_weight:
                nabla_Weight.append([np.zeros(w.shape) for w in obj.Weight])
        
        nabla_Biase  = []
        for obj in self.Layer:
            if obj.bool_biase:
                nabla_Biase.append([np.zeros(b.shape) for b in obj.Biase])
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for  delta_Weight, delta_Biase, cost in executor.map(self.backpropagation, [X for X, _ in Data], [Y for _, Y in Data]):
                nabla_Weight = [[nw_+dw_ for nw_, dw_ in zip(nw, dw)] for nw, dw in zip(nabla_Weight, delta_Weight)]
                nabla_Biase  = [[nb_+db_ for nb_, db_ in zip(nb, db)] for nb, db in zip(nabla_Biase, delta_Biase)]
                self.cost_learning += cost
        
        if self.cm_bool:
            #on utilise la methode du moment classique
            #on update la valeur du moment avec la nouvelle estimation du gradien
            self.momentum_Weight = [[self.m_constant*mw_ - (Epsilon/len(Data))*nw_ for mw_, nw_ in zip(mw, nw)] for mw, nw in zip(self.momentum_Weight, nabla_Weight)]
            self.momentum_Biase = [[self.m_constant*mb_ - (Epsilon/len(Data))*nb_ for mb_, nb_ in zip(mb, nb)] for mb, nb in zip(self.momentum_Biase, nabla_Biase)]
            #on update les valeur des poids
            i_w = 0     #iteration pour les poids
            i_b = 0     #iteration pour les biases
            for obj in self.Layer:
                if obj.bool_weight:
                    obj.Weight = [w + mw for w, mw in zip(obj.Weight, self.momentum_Weight[i_w])]
                    i_w += 1
                if obj.bool_biase:
                    obj.Biase = [b + mb for b, mb in zip(obj.Biase, self.momentum_Biase[i_b])]
                    i_b += 1
            
        elif self.nag_bool:
            #on utilise la methode de l'acceleration du gradient de nestorov
            #on sauvegarde la derniere valeur du gradien
            momentum_W_prev = self.momentum_Weight
            momentum_B_prev = self.momentum_Biase
            #on update la valeur du moment avec la nouvelle estimation du gradien
            self.momentum_Weight = [[self.m_constant*mw_ - (Epsilon/len(Data))*nw_ for mw_, nw_ in zip(mw, nw)] for mw, nw in zip(self.momentum_Weight, nabla_Weight)]
            self.momentum_Biase  = [[self.m_constant*mb_ - (Epsilon/len(Data))*nb_ for mb_, nb_ in zip(mb, nb)] for mb, nb in zip(self.momentum_Biase, nabla_Biase)]
            #on update les valeur des poids
            i_w = 0     #iteration pour les poids
            i_b = 0     #iteration pour les biases
            for obj in self.Layer:
                if obj.bool_weight:
                    obj.Weight = [w - self.m_constant*mw_prev + (1+self.m_constant)*mw for w, mw, mw_prev in zip(obj.Weight, self.momentum_Weight[i_w], momentum_W_prev[i_w])]
                    i_w += 1
                if obj.bool_biase:
                    obj.Biase = [b - self.m_constant*mb_prev + (1+self.m_constant)*mb for b, mb, mb_prev in zip(obj.Biase, self.momentum_Biase[i_b], momentum_B_prev[i_b])]
                    i_b += 1
        else:
            #on utilise la methode de descente de gradient classique
            #on update les valeur des poids
            i_w = 0     #iteration pour les poids
            i_b = 0     #iteration pour les biases
            for obj in self.Layer:
                if obj.bool_weight:
                    obj.Weight = [w - (Epsilon/len(Data))*nw for w, nw in zip(obj.Weight, nabla_Weight[i_w])]
                    i_w += 1
                if obj.bool_biase:
                    obj.Biase = [b - (Epsilon/len(Data))*nb for b, nb in zip(obj.Biase, nabla_Biase[i_b])]
                    i_b += 1
            
    
    def backpropagation(self, X, Y_expected):
        """Permet de calculé le gradient des poids et biase
            Inputs:
                -X          : valeur d'entré (narray, size=(1, .), type=float)
                -Y_expected : valeur attendue du reseau neuronal (narray, size=(1, .), type=float)"""
        delta_weight = []
        delta_biase  = []
        
        activations = [X] #liste des etage d'activation
        H           = X #activation en cour d'utilisation
        for obj in self.Layer:
            H = obj.feedforward(H) #calcule du la prochaine activation
            activations.append(H)
        
        #on gagnie tu temps en utilisant les calcule deja fait pour evalué le cost
        cost = self.cost_fonction.feedforward(H, Y_expected)
        
        X_prime  = self.cost_fonction.gradient(activations[-1], Y_expected) # calcule de la derivé du gradient
        
        for i in range(len(self.Layer)-1, 0, -1):
            #on parcoure la liste des etages a l'envere d'ou la propagation retour
            if self.Layer[i].bool_weight:
                X_prime, delta = self.Layer[i].gradient_activation(activations[i], X_prime), self.Layer[i].gradient_weight(activations[i], X_prime)
                
                if self.Layer[i].bool_biase:
                    delta_weight.append(delta[0])
                    delta_biase.append(delta[1])
                else:
                    delta_weight.append(delta)         
            else:
                X_prime                = self.Layer[i].gradient_activation(activations[i], X_prime)
        
        if self.Layer[0].bool_weight:
            delta = self.Layer[0].gradient_weight(activations[0], X_prime)
            if self.Layer[0].bool_biase:
                delta_weight.append(delta[0])
                delta_biase.append(delta[1])
            else:
                delta_weight.append(delta)
            
        return (delta_weight[::-1], delta_biase[::-1], cost)