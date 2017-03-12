import math
from matplotlib.pyplot import title, plot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn # improves matplotlib look and feel
import sys
import time

'''
Results visualization. 
'''

GLOBAL_TIME = 1.5

def load_file(path):
    return np.loadtxt(path, delimiter=',')

def plot_single_graph(path, chart_title, filename):
    print 'Graphing single ' + chart_title + '\n'
    values = load_file(path)
    print 'Max value ' + str(np.amax(values)) + ' and index ' + str(np.argmax(values)) + '\n'  
    plt.plot(values, 'b')
    plt.axvline(np.argmax(values), color ='black')
    plt.title(chart_title)
    plt.savefig(filename, format='png')
    plt.show(block=False)
    time.sleep(GLOBAL_TIME)
    plt.close()

def plot_single_graph_std(path, chart_title, filename):
    print 'Graphing single ' + chart_title + ' with standard deviation \n'
    values = load_file(path)
    print 'Max value ' + str(np.amax(values[0,:])) + ' and index ' + str(np.argmax(values[0,:])) + '\n'
    plt.plot(values[0,:], 'b')
    plt.errorbar(np.arange(values.shape[1]), values[0,:], yerr=values[1,:], linestyle='None', color ='b')
    plt.axvline(np.argmax(values), color ='black')
    plt.title(chart_title)
    plt.savefig(filename, format='png')
    plt.show(block=False)
    time.sleep(GLOBAL_TIME)
    plt.close()

def plot_double_graph(evalPath, trainPath, chart_title, filename):
    print 'Graphing double' + chart_title + '\n'
    values1 = load_file(evalPath)
    values2 = load_file(trainPath)
    print 'Evaluation max value ' + str(np.amax(values1)) + ' and index ' + str(np.argmax(values1)) + '\n'
    print 'Training max value ' + str(np.amax(values2)) + ' and index ' + str(np.argmax(values2)) + '\n'
    plt.plot(values1, 'b', label='Evaluation')
    plt.plot(values2, 'r', label='Training')
    plt.axvline(np.argmax(values1), color ='black')
    plt.legend(loc='upper center')
    plt.title(chart_title)
    plt.savefig(filename, format='png')
    plt.show(block=False)
    time.sleep(GLOBAL_TIME)
    plt.close()

def plot_double_graph_std(evalPath, trainPath, chart_title, filename):
    print 'Graphing double' + chart_title + ' with standard deviation \n'
    values1 = load_file(evalPath)
    values2 = load_file(trainPath)
    print(values1)
    print 'Evaluation max value ' + str(np.amax(values1[0,:])) + ' and index ' + str(np.argmax(values1[0,:])) + '\n'
    print 'Training max value ' + str(np.amax(values2[0,:])) + ' and index ' + str(np.argmax(values2[0,:])) + '\n'
    plt.plot(values1[0,:], 'b', label='Evaluation')
    plt.plot(values2[0,:], 'r', label='Training')
    plt.errorbar(np.arange(values1.shape[1]), values1[0,:], yerr=values1[1,:], linestyle='None', color ='b')
    plt.errorbar(np.arange(values2.shape[1]), values2[0,:], yerr=values2[1,:], linestyle='None', color ='r')
    plt.axvline(np.argmax(values1[0,:]), color ='black')
    plt.legend(loc='upper center')
    plt.title(chart_title)
    plt.savefig(filename, format='png')
    plt.show(block=False)
    time.sleep(GLOBAL_TIME)
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Missing arguments; Need to specify at least plot_type, plot_title, file to write to and one file containing the data'
        sys.exit(1)
    plot_type = sys.argv[1]
    if (len(sys.argv) < 6 and (plot_type == 'double' or plot_type == 'doubleWithStd')):
        print 'Missing arguments for double case; Need to specify plot_type, plot_title, file to write to and two files containing the two datasets'
        sys.exit(1)

    plot_type = sys.argv[1]
    plot_title = sys.argv[2]
    filename = sys.argv[3]
    path = sys.argv[4]
    
    if len(sys.argv) == 6:
    	path2 = sys.argv[5]

    if plot_type == 'single':
        plot_single_graph(path, plot_title, filename)
    elif plot_type == 'singleWithStd':
        plot_single_graph_std(path, plot_title, filename)
    elif plot_type == 'double':
        plot_double_graph(path, path2, plot_title, filename)
    elif plot_type == 'doubleWithStd':
        plot_double_graph_std(path, path2, plot_title, filename)
