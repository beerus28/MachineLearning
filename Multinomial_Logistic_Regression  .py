
# coding: utf-8

# The tags are in Begin-Inside-Outside (BIO) format. 
# Tags starting with B indicating the beginning of a piece of information, 
# tags beginning with I indicating a continuation of a previous type of information, and 
# O tags indicating words outside of any information chunk. 
# For example, in the sentence below, Newark is the departure city (fromloc.city_name), Los Angeles is the destination (toloc.city_name), and the user is requesting flights for Wednesday (depart_date.day_name). In this homework, you will treat the BIO tags as arbitrary labels for each word.
# 
# what    O
# flights O
# from    O
# newark  B-fromloc.city_name
# to      O
# los     B-toloc.city_name
# angeles I-toloc.city_name
# on      O
# wed     B-depart_date.day_name
#                   
# 

# In[1]:


import csv
import numpy as np
import math


#  a few HINTS to get you started:-
#  
# • Write a helper function that produces a sparse representation of the data (see Section 2.3.4 on feature engineering for details on how to do this).
# 
# 

# It is recommended that you create a separate parsing function for each model that 
# reads in the input .tsv file and 
# outputs a label array and a feature array. 
# 
# Each feature (wt−1, wt, and wt+1) should use a "one-hot encoding scheme". 
# In this scheme, each element of the feature (or label) represents a possible word (or label). All elements of the vector should be zero except for the element corresponding to the word assigned to the feature. 
# 
# For example, if the current word, wt, is “Boston”, and “Boston” corresponds to the ith element in our list of possible features, wt will be a vector with the ith position being 1 and everywhere else being 0. 
# 
# Alternatively, one can sparsely represent the feature value by making wt a 1-dimensional integer variable and assigning wt an index corresponding to the ith word in our list of possible features. This index could them be used to extract parameters from your model weight matrix. The labels can also be encoded using a similar format.

# In[2]:


#New
def create_table_A2_():
    word_w_t = [] 
    tag_y_t = []
    position_t = []
    count = 0
    index_i = []
    label_y_i = []
    feature_x_i = []
    count = 0
    #with open('handout/toydata/toytrain.tsv', 'r') as tsvfile:
    with open('handout/largedata/train.tsv', 'r') as tsvfile:

        for line in tsvfile:
            row = line.split()    # WHAT DOES THIS DO ???  
            if row:
                label_y_i.append(row[1])
                #feature_x_i.append("cur:" + row[0] + " =1")
                feature_x_i.append(row[0])

                count += 1
                index_i.append(count)
    
    
    label_y_i_dict = {}
    feature_x_i_dict = {}
    
    for i in feature_x_i:
        if i not in feature_x_i_dict.keys():
            #feature_x_i_dict[i] = 1
            feature_x_i_dict.update({i : 1})
        else:
            continue
            
    for j in label_y_i:
        if j not in label_y_i_dict.keys():
            label_y_i_dict[j] = 1
        else:
            continue
    
    #print(feature_x_i_dict, label_y_i_dict)
    
    k = len(label_y_i_dict) #rows # rows/depth of theta matrix 
    m = len(feature_x_i_dict)   # columns # lenght of theta matrix
    k_n = len(label_y_i)
    #feature_set_x_i = list(set(feature_x_i)) 
    
    print("k = ", k, ", m = ", m, "k_n = ", k_n)
    
    theta = np.zeros([k, m])
    
    model_1_input = np.zeros([k_n, m])
    
    #print(feature_x_i_dict.keys())
    key_pos = list(feature_x_i_dict.keys())
    weights_Length = len(key_pos)
    print(weights_Length, key_pos)
    Model_1_Feature_One_Hot_Positions = [key_pos.index(x) for x in feature_x_i]
    #print(Model_1_Feature_One_Hot_Positions) # training example features

    key_pos2 = list(label_y_i_dict.keys())
    weights_Length2 = len(key_pos2)
    print(weights_Length2, key_pos2)
    Model_1_label_One_Hot_Positions = [key_pos2.index(x) for x in label_y_i]
    #print(Model_1_label_One_Hot_Positions) # training example features

    return Model_1_Feature_One_Hot_Positions, feature_x_i, feature_x_i_dict, key_pos, weights_Length, k, m, Model_1_label_One_Hot_Positions, label_y_i_dict, key_pos2, weights_Length2;
    
    
create_table_A2_()


# In[3]:




def indicator_fn(y_sup_i, k):  #???
    if y_sup_i == k:
        return 1
    else:
        return 0


#calculate exp denominator

def den_calc(x_i, theta):
    
    den_sum = 0.0 
    for j in theta:
        den_sum += np.exp(np.matmul(np.transpose(j), x_i))#theta[label_y_i.index(j)]), x_i)) # x_sup_i is ith eg in whole X        # numpy.transpose(a, axes=None) #numpy.matmul(a, b, out=None)
                        # sigma j=1 to k exp((θ^j)^⊤ x^i)
        #print(den_sum) 
    return den_sum

# GET LOG LIKELIHOOD 

def get_log_likelihod():
    
    Model_1_Feature_One_Hot_Positions, feature_x_i, feature_x_i_dict, key_pos, weights_Length, k, m, Model_1_label_One_Hot_Positions, label_y_i_dict, key_pos2, weights_Length2 = create_table_A2_()
    
    theta = np.zeros([k, m+1])
    theta[:, 0] = 1
    learn_rate = 0.5
    Total = 0 
    for r in range(2):
        
        for i in range(len(Model_1_Feature_One_Hot_Positions)):

            #if(Model == 1):
                Feature_Vector = np.zeros([weights_Length + 1, 1])  # extra 1 t ake it 2-D 
                Feature_Vector[0] = 1 # corresponding to bias term 
                Feature_Vector[Model_1_Feature_One_Hot_Positions[i] + 1] = 1

                label_vector = np.zeros([weights_Length2, 1])  # extra 1 t ake it 2-D 
                label_vector[Model_1_label_One_Hot_Positions[i]] = 1

                grad_sgd = np.zeros([k, m+1])
                grad_sgd[:, 0] = 1

                for j in range(len(key_pos2)):   # gradient calculated for all k before updating any individual theta_sup_k
                
                    indc_don = (indicator_fn(Model_1_label_One_Hot_Positions[i], j))
                    num = (np.exp(np.matmul(np.transpose(theta[j]), Feature_Vector)))
                    deno = den_calc(Feature_Vector, theta)
                    send_val = num/deno
                    fin_left_val = indc_don - (num / deno)
                    #print(len(Feature_Vector))
                    grad_sgd[j] = -1 * fin_left_val * Feature_Vector.T                                                                         # convert matrix to num, in power of e

                theta = np.subtract(theta, (learn_rate*grad_sgd))      # one “stochastic gradient” step in O(1) time
                
        r += 1
    return theta 
    
get_log_likelihod()



# In[5]:


# PREDICT

def den_calc_p(x_i):
    den_sum = 0.0 
    for j in theta:
        den_sum += np.exp(np.matmul(np.transpose(j), x_i))#theta[label_y_i.index(j)]), x_i)) # x_sup_i is ith eg in whole X        # numpy.transpose(a, axes=None) #numpy.matmul(a, b, out=None)
        #print(den_sum)             # sigma j=1 to k exp((θ^j)^⊤ x^i) 
    return den_sum

test_x_feature = []
y_label = []


Model_1_Feature_One_Hot_Positions, feature_x_i, feature_x_i_dict, key_pos, weights_Length, k, m, Model_1_label_One_Hot_Positions, label_y_i_dict, key_pos2, weights_Length2 = create_table_A2_()

theta = get_log_likelihod()

#with open('handout/toydata/toyvalidation.tsv', 'r') as tsvfile2:
with open('handout/largedata/test.tsv', 'r') as tsvfile2:
        
        for line in tsvfile2:
            
            row = line.split()
            count = 0
            if row:
                test_x_feature.append(row[0])
                #index_x_feat = feature_set_x_i.index(row[0])
                
        #print(weights_Length, key_pos)
        Model_1_test_Feature_One_Hot_Positions = [key_pos.index(x) for x in test_x_feature]
        #print(Model_1_Feature_One_Hot_Positions) # test example features

        for i in range(len(Model_1_test_Feature_One_Hot_Positions)):

            #if(Model == 1):
                Feature_Vector = np.zeros([weights_Length + 1, 1])  # extra 1 t ake it 2-D 
                Feature_Vector[0] = 1 # corresponding to bias term 
                Feature_Vector[Model_1_test_Feature_One_Hot_Positions[i] + 1] = 1

                label_vector = np.zeros([weights_Length2, 1])  # extra 1 t ake it 2-D 
                label_vector[Model_1_label_One_Hot_Positions[i]] = 1

                grad_sgd = np.zeros([k, m+1])
                grad_sgd[:, 0] = 1

                max_prob = -1.0
                result_label = ""
                for j in range(len(key_pos2)):   # gradient calculated for all k before updating any individual theta_sup_k
                
                    #indc_don = (indicator_fn(Model_1_label_One_Hot_Positions[i], j))
                    num = (np.exp(np.matmul(np.transpose(theta[j]), Feature_Vector)))
                    deno = den_calc(Feature_Vector, theta)
                    send_val = num/deno
                    #fin_left_val = indc_don - (num / deno)
                    #print(len(Feature_Vector))
                    if send_val > max_prob:
                        
                        max_prob = send_val
                        result_label = ""
                        result_label = j
                        reslt = list(label_y_i_dict.keys())[j]
                print(reslt, result_label, max_prob) 

                
                
                
                
                
                
                


# In[ ]:


#table A1:
def create_table_A1():

    word_w_t = [] 
    tag_y_t = []
    position_t = []
    count = 0
    with open('handout/toydata/toytrain.tsv', 'r') as tsvfile:
        for line in tsvfile:
            row = line.split()    # WHAT DOES THIS DO ???  
            if row:
                word_w_t.append(row[0])
                tag_y_t.append(row[1])
                count += 1
                position_t.append(count)
            else:
                count = 0

    table_A1 = np.array([word_w_t, tag_y_t]) # FEATURE IS ROW, N LABEL IS ANOTHER ROW
    # step 1: x = list(zip(position_t, word_w_t, tag_y_t)  # step 2: np_pair = np.asarray(x)     
    np_table_A1 = np.asarray(list(zip(position_t, word_w_t, tag_y_t))) # EACH ROW IS A "FEATURE-LABEL PAIR"
    #print(np_table_A1)
    return np_table_A1

create_table_A1()    # CALLING
# The ith row of this file will be used to construct the ith training example using either Model 1 features or Model 2 features



# In[ ]:


#table A3:
def create_table_A3():
    # do we delete repeating words???
    
    index_i = []
    label_y_i = []
    feature_x_i = []
    count = 0
    with open('handout/toydata/toytrain.tsv', 'r') as tsvfile:
        for line in tsvfile:
            row = line.split()    # WHAT DOES THIS DO ???  
            if row:
                label_y_i.append(row[1])
                feature_x_i.append("cur:" + row[0] + " =1")
                count += 1
                index_i.append(count)
    
    np_table_A3 = np.asarray(list(zip(index_i, label_y_i, feature_x_i))) # EACH ROW IS A "LABEL-FEATURE PAIR"
    print("INDEX (i)" + "| LABEL {tag} (y^i)" + " | FEATURE {word} (x^i)" )
    #print(np_table_A3)
    
    k = len(set(label_y_i))
    m = len(set(feature_x_i))
    print("k = ", k, ", m = ", m)
    return np_table_A3   # return statements to get access to these table arrays

    
create_table_A3()
# Each feature vector is now represented by a map from the string name of the feature (e.g. cur:angeles) 
# to its value (e.g. 1).

#The ith row corresponds to the ith training example.



# • Write a function that takes a single SGD step on the ith training example. Such a function should take as input the model parameters, the learning rate, and the features and label for the ith training example. It should update the model parameters in place by taking one stochastic gradient step.
# 
# 

# In[ ]:


#input : the "model parameters", the "learning rate = 0.5", and the "features & label" for the "ith training example".
#output : update the model parameters in place by taking one stochastic gradient step

index_i = []
label_y_i = []
feature_x_i = []
count = 0
learn_rate = 0.5

feature_vectors_tb_A2, feature_set_x_i, label_y_i = create_table_A2()        

k_N = 108      #k = no. of rows = no. of possible y values
m = 61

# indicator function
def indicator_fn(y_sup_i, k):  #???
    if y_sup_i == k:
        return 1
    else:
        return 0
    
theta = np.zeros([k_N, m])
grad_sgd = np.zeros([k_N, m])

#calculate exp denominator
def den_calc(x_i):
    
    den_sum = 0.0 
    for j in theta:
        den_sum += np.exp(np.matmul(np.transpose(j), x_i))#theta[label_y_i.index(j)]), x_i)) # x_sup_i is ith eg in whole X        # numpy.transpose(a, axes=None) #numpy.matmul(a, b, out=None)
                        # sigma j=1 to k exp((θ^j)^⊤ x^i)
        #print(den_sum) 
    return den_sum
    
#compute the example-specific gradient and
#grad_sgd_theta_k = 0.0
for i in range(k_N):  # iterating through training examples
    for j in label_y_i:   # gradient calculated for all k before updating any individual theta_sup_k
            
        grad_sgd[label_y_i.index(j)] += (indicator_fn(label_y_i[i], j) - (np.exp(np.matmul(np.transpose(theta[label_y_i.index(j)]), feature_vectors_tb_A2[i])) / den_calc(feature_vectors_tb_A2[i])))*feature_vectors_tb_A2[i]                                                                         # convert matrix to num, in power of e
        
    theta = np.subtract(theta, (-1*learn_rate*grad_sgd))      # one “stochastic gradient” step in O(1) time
    



# In[ ]:


print(theta) 


# In[ ]:


# #test
# def den_calc_p(x_i):
#     den_sum = 0.0 
#     for j in theta:
#         den_sum += np.exp(np.matmul(np.transpose(j), x_i))#theta[label_y_i.index(j)]), x_i)) # x_sup_i is ith eg in whole X        # numpy.transpose(a, axes=None) #numpy.matmul(a, b, out=None)
#         #print(den_sum)             # sigma j=1 to k exp((θ^j)^⊤ x^i) 
#     return den_sum

# feature_vectors_tb_A2, feature_set_x_i, label_y_i = create_table_A2()        
# x_feature = []
# y_label = []
# with open('handout/toydata/toytest.tsv', 'r') as tsvfile2:
        
#         for line in tsvfile2:
            
#             row = line.split()
#             count = 0
#             if row:
#                 x_feature.append(row[0])
#                 index_x_feat = feature_set_x_i.index("cur:" + row[0] + " =1")
# #                 temp = row[0]
# #                 y_label.append(row[1])
                
#                 result_prob = 0.0      #label_prob_list = []
#                 #for each test input:  # iterating through training examples
#                 max_prob = -1.0
#                 result_label = ""
#                 print(feature_set_x_i[index_x_feat])
#                 for j in label_y_i:   # gradient calculated for all k before updating any individual theta_sup_k
#                     result_prob = (np.exp(np.matmul(np.transpose(theta[label_y_i.index(j)]), feature_vectors_tb_A2[index_x_feat])) / den_calc_p(feature_vectors_tb_A2[index_x_feat]))

#                     if result_prob > max_prob:
#                         max_prob = result_prob
#                         result_label = ""
#                         result_label = j
#                 print(max_prob, result_label) 

    


# • Write a function that takes in a set of features, labels, and model parameters and then outputs the error (percentage of labels incorrectly predicted). You can also write a separate function that takes the same inputs and outputs the negative log-likelihood of the regression model.
# 
# • Be sure to handle boundary cases as specified in (see Section 2.3.4)

# We have provided you with two subsets of the ATIS data set. Each one is divided into a training, a validation, and a test data set. The toy data set (toytrain.tsv, toyvalidation.tsv, and toytest.tsv) is a small data set that can be used while debugging your code. We have included the reference output files for this toy data set (see directory toyoutput/). We have also included a larger data set without reference outputs (train.csv, validation.csv, test.csv). This data set can be used to ensure that your code runs fast enough to pass the autograder tests. Your code should be able to perform one pass through all of the data in less than one minute for each of the models: one minute for Model 1 and one minute for Model 2.
# 
# 
# 

# The files are in tab-separated-value (.tsv) format. This is identical to a comma-separated-value (.csv) format except that instead of separating columns with commas, we separate them with a tab character, \t. Each row is ended by a Unix style line ending, \n. The first column always contains the word and the second column the tag.
