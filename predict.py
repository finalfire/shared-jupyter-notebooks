import numpy as np
import builder
from keras.models import load_model
from vis.visualization import visualize_activation
from scipy.stats import normaltest

def prediction_with_important_edges(brain_net, graph_adjacency, n_nodes=84, kth=100):

    # make prediction for a single graph
    pred_bn, confidences = brain_net.predict(graph_adjacency)

    print('Computing partial derivatives for each samples')
    print('This operation requires time...')

    # graph_adjacency = np.expand_dims(graph_adjacency, axis=0)

    # compute for each edge the partial derivatives according to Simonyan et al (2013)
    heatmap = visualize_activation(brain_net.model, layer_idx=-1, filter_indices=0, seed_input=graph_adjacency)

    edges = list()

    # getting the triu of the matrix
    triu = np.triu(heatmap)
    n,m = triu.shape
    for i in range(n):
        for j in range(i+1, m):
            edges.append((i+1,j+1,triu[i][j]))

    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    print('done.')
    
    return confidences, pred_bn, edges


def prediction_with_important(brain_net, graph_adjacency, n_nodes=84, kth=100):

    # make prediction for a single graph
    pred_bn, confidences = brain_net.predict(graph_adjacency)

    print('Computing partial derivatives for each samples')
    print('This operation requires time...')

    # graph_adjacency = np.expand_dims(graph_adjacency, axis=0)

    # compute for each edge the partial derivatives according to Simonyan et al (2013)
    heatmap = visualize_activation(brain_net.model, layer_idx=-1, filter_indices=0, seed_input=graph_adjacency)

    # compute the importance of each node by summing importance of edges
    important_nodes = np.sum(abs(heatmap), axis=0)

    # select the nodes belonging to the percentile-th percentile
    #most_important = np.argwhere(important_nodes >= np.percentile(important_nodes, percentile))
    
    # sorting for getting the first k% most important
    vip = [(i, value) for i, value in enumerate(important_nodes)]
    k_vip = sorted(vip, reverse=True, key=lambda x: x[1])[n_nodes - int((kth/100) * n_nodes):]
    print(k_vip)

    #k_vip = list(map(lambda x: x[0]+1, k_vip))

    print('done.')

    #return confidences, pred_bn, most_important
    return confidences, pred_bn, k_vip

def prediction(brain_net, graph_adjacency, n_nodes=84, kth=80):

    # make prediction for a single graph
    pred_bn, confidences = brain_net.predict(graph_adjacency)

    #print('Computing partial derivatives for each samples')
    #print('This operation requires time...')

    # graph_adjacency = np.expand_dims(graph_adjacency, axis=0)

    # compute for each edge the partial derivatives according to Simonyan et al (2013)
    #heatmap = visualize_activation(brain_net.model, layer_idx=-1, filter_indices=0, seed_input=graph_adjacency)

    # compute the importance of each node by summing importance of edges
    #important_nodes = np.sum(abs(heatmap), axis=0)

    # select the nodes belonging to the percentile-th percentile
    #most_important = np.argwhere(important_nodes >= np.percentile(important_nodes, percentile))
    
    # sorting for getting the first k% most important
    #vip = [(i, value) for i, value in enumerate(important_nodes)]
    #k_vip = sorted(vip, reverse=True, key=lambda x: x[1])[n_nodes - int((kth/100) * n_nodes):]

    #k_vip = list(map(lambda x: x[0]+1, k_vip))

    #print('done.')

    #return confidences, pred_bn, most_important
    return confidences, pred_bn #, k_vip


def build_model():

    brain_net = builder.BRAIN_net()
    brain_net.build_model(load_weights=True)

    return brain_net
