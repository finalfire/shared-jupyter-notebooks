import predict
import numpy as np
import os
import csv

_STAGES = ('CIS', 'RR', 'PP', 'SP')
_HEADER_CSV = ('Stage', 'Patient', 'N_Edges_Removed', 'CIS', 'PP', 'RR', 'SP', 'OUTCLASS')
_MAX_N_EDGE_TO_REMOVE = 300

def main():
    # start model
    model = predict.build_model()
    
    # results array
    results = list()

    for stage in _STAGES:
        # per ogni paziente
        for patient in os.listdir('Matrices/' + stage):

            print(stage, patient, '...')

            # path matrice paziente
            patient_path = 'Matrices/{}/{}'.format(stage, patient)

            # leggo la matrice
            original_adj = np.genfromtxt(patient_path, delimiter=' ')
            original_adj = original_adj + original_adj.T
            np.fill_diagonal(original_adj, 0)

            # predizione
            data, label = predict.prediction(model, original_adj)
            c1_value, c2_value, c3_value, c4_value = data[0]

            # salvo ad indice 0
            results.append((
                stage, patient, 0,
                c1_value, c2_value, c3_value, c4_value, label
            ))

            # leggo gli archi importanti (ordine per importanza decrescente)
            important_edges = list()
            with open('ImportantEdges/' + stage + '/' + patient) as impedges_file:
                important_edges = impedges_file.read().rstrip().split(' ')
                important_edges = [list(map(int, x.split(','))) for x in important_edges]
            
            # per ogni arco importante in i = 1..300
            for n_edge in range(0, 300):
                # azzero questo arco nella matrice
                x,y = important_edges[n_edge][0]-1, important_edges[n_edge][1]-1
                original_adj[x][y] = 0
                original_adj[y][x] = 0

                # predizione
                data, label = predict.prediction(model, original_adj)
                c1_value, c2_value, c3_value, c4_value = data[0]

                # salvo ad indice i
                results.append((
                    stage, patient, n_edge,
                    c1_value, c2_value, c3_value, c4_value, label
                ))
    
    # appendo al csv
    with open('results_brutefor.csv', 'w') as f_out:
        w_to_csv = csv.writer(f_out, delimiter=';')
        w_to_csv.writerow(_HEADER_CSV)
        for row in results:
            w_to_csv.writerow(row)

if __name__ == "__main__":
    main()