import numpy as np
import GPy
from pylab import *

def checkModelAccuracy(dynModel, xtest, ytest, isCorregional):
    if(isCorregional == 1):
        newX = np.hstack([xtest,0*np.ones_like(xtest[:,0]).reshape(xtest.shape[0],1)])
        noise_dict = {'output_index':newX[:,xtest.shape[1]:].astype(int)}
        results = dynModel.predict(newX,Y_metadata=noise_dict)
        ypred = results[0]
        sGP = results[1]
        for ii in range(0,ytest.shape[1]):
            print("output: "+ str(ii))
            rsqTrain, maeTrain, rsqAdjTrain = evaluateGoodnessOfFit(xtest,ytest[:,ii],ypred[:,ii],1)
            print("Rsq train Gaussian Processes Regression = " + str(rsqTrain))
            print("Rsq Adjusted train Gaussian Processes Regression = " + str(rsqAdjTrain))
            print("MAE train Gaussian Processes Regression = " + str(maeTrain))
            print("------------------------------------------------------")
            
#            t = np.array(range(0,len(ytest)))
#            plt.plot(t, ytest[:,ii], 'b')
#            plt.errorbar(t, ypred[:,ii], sGP, color = 'r', ecolor='r')
#            plt.show()
#            plt.close()
    else:
        results = dynModel.predict(xtest)
        ypred = results[0]
        sGP = results[1]
         
        rsqTrain, maeTrain, rsqAdjTrain = evaluateGoodnessOfFit(xtest,ytest,ypred,1)
        print("Rsq train Gaussian Processes Regression = " + str(rsqTrain))
        print("Rsq Adjusted train Gaussian Processes Regression = " + str(rsqAdjTrain))
        print("MAE train Gaussian Processes Regression = " + str(maeTrain))
        
#        t = np.array(range(0,len(ytest)))
#        plt.plot(t, ytest, 'b')
#        plt.errorbar(t, ypred, sGP, color = 'r', ecolor='r')
#        plt.show()
#        plt.close()

 
def evaluateGoodnessOfFit(x,y,ypred,flat):
    """ Reporting --- Calculate Rsquared and MAE """
    print(y.shape[0])
    print(x.shape[1])
    y_hat = np.mean(y)
    SStot = np.sum(np.power(y - y_hat,2))
    if(flat==1):
        SSres = np.sum(np.power(y - ypred.flatten(),2))
    else:
        SSres = np.sum(np.power(y - ypred,2))
    if(SStot == 0):
        rsq = 1
    else:
        rsq = 1 - SSres/SStot
    if(flat==1):
        mae = np.sum(np.abs(y - ypred.flatten()))/ypred.shape[0]
    else:
        mae = np.sum(np.abs(y - ypred))/ypred.shape[0]
    rsqAdj = 1 - (1-rsq) * (y.shape[0]-1) / (y.shape[0]-x.shape[1]-1)
    
    return rsq, mae, rsqAdj

