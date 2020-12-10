"""
This is a class which implements various metrics with normalization for training. It includes a squared error loss
taking into acount the circular nature of angle, errors on pte, eta, phi, and E seperate, and dR loss
"""

import tensorflow as tf
from math import pi

class metrics(object):
    def __init__(self, meanPT=0, sdPT=1, meanEta=0, sdEta=1, meanPhi=0, sdPhi=1):
        """
        Constructor, takes normalization information.
        """
        self.meanPT = meanPT
        self.sdPT = sdPT
        self.meanEta = meanEta
        self.sdEta = sdEta
        self.meanPhi = meanPhi
        self.sdPhi = sdPhi
    
    def custom_loss(self, y_true, y_pred):
        """
        Squared error taking into account circular nature of pi. 
        """
        val = tf.minimum(tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]),2),tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]+2*pi),2))
        val = tf.minimum(val,tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]-2*pi),2))
        val1 = tf.reduce_mean(val)
        val2 = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,0], y_pred[:,0]),2))
        val3 = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,1], y_pred[:,1]),2))
        val4 = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,3], y_pred[:,3]),2))

        return([val1, val2, val3, val4])

    def pt_error(self, y_true, y_pred):
        """Squared error of pt prediction"""
        val = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,0], y_pred[:,0]),2))

        return(val)

    def eta_error(self, y_true, y_pred):
        """Squared error of eta prediction"""
        val = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,1], y_pred[:,1]),2))

        return(val)

    def phi_error(self, y_true, y_pred):
        """Squared error of phi prediction with circular nature"""
        val = tf.minimum(tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]),2),tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]+2*pi),2))
        val = tf.minimum(val,tf.math.pow(tf.math.subtract(y_true[:,2], y_pred[:,2]-2*pi),2))
        val = tf.reduce_mean(val)
        return(val)

    def e_error(self, y_true, y_pred):
        """Squared error of e prediction"""
        val = tf.reduce_mean(tf.math.pow(tf.math.subtract(y_true[:,3], y_pred[:,3]),2))

        return(val)

    def dR(self, y_true, y_pred):
        """dR error"""
        truePT=tf.math.add(tf.math.multiply(y_true[:,0], self.sdPT), self.meanPT)
        trueEta=tf.math.add(tf.math.multiply(y_true[:,1], self.sdEta), self.meanEta)
        truePhi=tf.math.add(tf.math.multiply(y_true[:,1], self.sdPhi), self.meanPhi)

        predPT=tf.math.add(tf.math.multiply(y_pred[:,0], self.sdPT), self.meanPT)
        predEta=tf.math.add(tf.math.multiply(y_pred[:,1], self.sdEta), self.meanEta)
        predPhi=tf.math.add(tf.math.multiply(y_pred[:,1], self.sdPhi), self.meanPhi)


        raw_dphi = truePhi - predPhi;
        dphi = tf.where(tf.abs(raw_dphi)<pi, raw_dphi, raw_dphi-2*pi*tf.math.round(raw_dphi/2*pi))

        deta = trueEta - predEta;
        val=tf.math.pow(tf.math.pow(deta,2)+tf.math.pow(dphi,2),0.5)
        return(tf.reduce_mean(val))
    
    def dRRaw(self, y_true, y_pred):
        """dR error, but not a mean. This just gives all the dR values."""
        
        truePT=tf.math.add(tf.math.multiply(y_true[:,0], self.sdPT), self.meanPT)
        trueEta=tf.math.add(tf.math.multiply(y_true[:,1], self.sdEta), self.meanEta)
        truePhi=tf.math.add(tf.math.multiply(y_true[:,1], self.sdPhi), self.meanPhi)

        predPT=tf.math.add(tf.math.multiply(y_pred[:,0], self.sdPT), self.meanPT)
        predEta=tf.math.add(tf.math.multiply(y_pred[:,1], self.sdEta), self.meanEta)
        predPhi=tf.math.add(tf.math.multiply(y_pred[:,1], self.sdPhi), self.meanPhi)


        raw_dphi = truePhi - predPhi;
        dphi = tf.where(tf.abs(raw_dphi)<pi, raw_dphi, raw_dphi-2*pi*tf.math.round(raw_dphi/2*pi))

        deta = trueEta - predEta;
        val=tf.math.pow(tf.math.pow(deta,2)+tf.math.pow(dphi,2),0.5)
        return(val)



    def dPT(self, y_true, y_pred):
        """predicted pt/true pt error"""
        truePT=tf.math.add(tf.math.multiply(y_true[:,0], self.sdPT), self.meanPT)

        predPT=tf.math.add(tf.math.multiply(y_pred[:,0], self.sdPT), self.meanPT)

        val=tf.math.divide(tf.exp(predPT),tf.exp(truePT))

        return(tf.reduce_mean(val))
