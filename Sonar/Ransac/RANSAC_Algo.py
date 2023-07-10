import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt

class LinearLeastSqaureModel:
    def fit(self, A, Y):
        A_transpose = A.transpose()
        ATA = A_transpose.dot(A)
        ATY = A_transpose.dot(Y)
        model = (np.linalg.inv(ATA)).dot(ATY) ## For a linear eq. AP = Y to solve a least sqaure problem,  P = (inverse(A'A))(A'Y) 
        return model
    
class RansacModel:

    
    def __init__(self, curve_fitting_model):
        self.curve_fitting_model = curve_fitting_model
    
    def fit(self, A, Y, num_sample, threshold):

        num_iterations = math.inf
        iterations_done = 0
        num_sample = 3

        max_inlier_count = 0
        best_model = None

        prob_outlier = 0.5
        desired_prob = 0.95

        total_data = np.column_stack((A, Y))  ## [ A | Y]
        data_size = len(total_data)

        # Adaptively determining the number of iterations
        while num_iterations > iterations_done:

            # shuffle the rows and take the first 'num_sample' rows as sample data
            np.random.shuffle(total_data)
            sample_data = total_data[:num_sample, :]
            
            estimated_model = self.curve_fitting_model.fit(sample_data[:,:-1], sample_data[:, -1:]) ## [a b c]

            # count the inliers within the threshold
            y_cap = A.dot(estimated_model)
            err = np.abs(Y - y_cap.T)
            inlier_count = np.count_nonzero(err < threshold)

            # check for the best model 
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model


            prob_outlier = 1 - inlier_count/data_size
            #print('# inliers:', inlier_count)
            #print('# prob_outlier:', prob_outlier)
            if prob_outlier!=1 and prob_outlier!=0:
                num_iterations = math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**num_sample)
            iterations_done = iterations_done + 1

            #print('# s:', iterations_done)
            #print('# n:', num_iterations)
            #print('# max_inlier_count: ', max_inlier_count)

        return best_model,max_inlier_count

def fit_walls(data): 
    x_values = np.array(data[0])
    y_values = np.array(data[1])

    """
    Based on the data the equation that is required is: bx + c = y
    So, we need to define matrices:
     1. A with columns [x  1]           Si on veux prendre des courbe suffit d'augmenter le degres de cette matrice (voir github RANSAC)
     2. P = [b c]*   (* = transpose)
     3. Y
    So that we will be finding the solution for the equation AP = Y
    """

    ## A = [ x  1] car on veux une droite
    A = np.stack((x_values, np.ones((len(x_values)), dtype = int)), axis = 1)
    threshold = np.std(y_values)/2  # this can be tuned to sd/3 or sd/5 for various curves and better consistent results as a result of random sampling
    
    # Instantiating the linear least sqaure model
    linear_ls_model = LinearLeastSqaureModel()
    linear_ls_model_estimate = linear_ls_model.fit(A, y_values)
    linear_model_y = A.dot(linear_ls_model_estimate)                    # Point des moindre carrer

    # Instantiating the ransac model
    ransac_model = RansacModel(linear_ls_model)
    ransac_model_estimate,Max_inliner = ransac_model.fit(A, y_values, 3, threshold)     # Equation de la droite [[b],[c]]
    ransac_model_y = A.dot(ransac_model_estimate)                       # Point du ransac

    return ransac_model_estimate,Max_inliner