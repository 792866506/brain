#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:00:52 2018

@author: al

but you need to run get_xgb_data.py  first
need three mat file  {'decision_values':prob})
                      {'actual_label':label})
                    {'predict_label':predict_label})
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io  as sio


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


'''
calculate each rate
'''
def cal_rate(result, num, thres):
	all_number = len(result[0])
	# print all_number
	TP = 0
	FP = 0
	FN = 0
	TN = 0
	for item in range(all_number):
		disease = result[0][item,num]
		if disease >= thres:
			disease = 1
		if disease == 1:
			if result[1][item,num] == 1:
				TP += 1
			else:
				FP += 1
		else:
			if result[1][item,num] == 0:
				TN += 1
			else:
				FN += 1
	# print TP+FP+TN+FN
	accracy = float(TP+TN) / float(all_number)
	if TP+FP == 0:
		precision = 0
	else:
		precision = float(TP) / float(TP+FP)
	TPR = float(TP) / float(TP+FN)
	TNR = float(TN) / float(FP+TN)
	FNR = float(FN) / float(TP+FN)
	FPR = float(FP) / float(FP+TN)
	# print accracy, precision, TPR, TNR, FNR, FPR
	return accracy, precision, TPR, TNR, FNR, FPR


'''
plot roc and calculate AUC/ERR, result: (prob, label) 
'''
prob_list = []
label_list= []
predict_label_list=[]
auc_list= []
path= 'shallow_aug'
for subject_id  in xrange(1,10):
    prob_list.append( sio.loadmat('/home/al/braindecode/result/'+path+\
                                  '/decision_values{:1d}.mat'
                    .format(subject_id))['decision_values'])
    label_list.append( sio.loadmat('/home/al/braindecode/result/'+path+\
                                   '/actual_label{:1d}.mat'
                    .format(subject_id))['actual_label'].reshape(288))
    predict_label_list.append(sio.loadmat('/home/al/braindecode/result/'+path+\
                                          '/predict_label{:1d}.mat'
                   .format(subject_id))['predict_label'].reshape(288))
prob = np.concatenate(prob_list)
actual_label = np.concatenate(label_list)
label = one_hot(actual_label-1)
'''
# when use matalb  PG_CUrve

predict_label =  np.concatenate(predict_label_list)
sio.savemat('/home/al/PG_Curve/decision_values.mat',
            { 'decision_values': prob })
sio.savemat('/home/al/PG_Curve/actual_label.mat',
            { 'actual_label': actual_label })
sio.savemat('/home/al/PG_Curve/predict_label.mat',
            { 'predict_label': predict_label })

'''

disease_class = ['left','right','foot','tongue']
style = ['r-','g-','b-','y-']
plt.figure()
#plt.ylim((0.1, 1.0))
my_y_ticks = np.arange(0, 1.1, 0.1)
plt.yticks(my_y_ticks)
for clss in range(len(disease_class)):
    threshold_vaule = sorted(prob[:,clss])
    threshold_num = len(threshold_vaule)
    accracy_array = np.zeros(threshold_num)
    precision_array = np.zeros(threshold_num)
    TPR_array = np.zeros(threshold_num)
    TNR_array = np.zeros(threshold_num)
    FNR_array = np.zeros(threshold_num)
    FPR_array = np.zeros(threshold_num)
	# calculate all the rates
    for thres in range(threshold_num):
        accracy, precision, TPR, TNR, FNR, FPR = cal_rate((prob,label), clss, threshold_vaule[thres])
        accracy_array[thres] = accracy
        precision_array[thres] = precision
        TPR_array[thres] = TPR
        TNR_array[thres] = TNR
        FNR_array[thres] = FNR
        FPR_array[thres] = FPR
    # print TPR_array
    # print FPR_array
    AUC = np.trapz(TPR_array, FPR_array)
    threshold = np.argmin(abs(FNR_array - FPR_array))
    EER = (FNR_array[threshold]+FPR_array[threshold])/2
    print ('%s threshold : %f' % (disease_class[clss],threshold))
    print ('%s accracy : %f' % (disease_class[clss],accracy_array[threshold]))
    print ('%s EER : %f AUC : %f' % (disease_class[clss],EER, -AUC))
    auc_list.append(-AUC)
    #plt.figure(figsize=(10,10))
    plt.plot((FPR_array), TPR_array, style[clss], label=disease_class[clss],linewidth=1.5)
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.legend(fontsize=13)
plt.show()
#plt.savefig('/home/al/braindecode/result/'+path+'/0{:1d}'\
#            .format(1),format='png',dpi=200)







