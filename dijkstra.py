import networkx as nx
import numpy as np
from time import time
from heapq import heappush, heappop

def rand_pos_graph(n, density):
    """ Build a random positive undirected graph for testing"""

    # Generate a random matrix with n rows and columns
    b = np.random.rand(n,n)
    # For each row
    for i in range(n):
        # For each column
        for j in range(i,n):
            # If the diagonal entry, set to 0
            if i == j:
                b[i,j] = 0
            # Else if a random draw is greater than the density function,
            # set the entry and its symmetric opposite to 0
            elif np.random.rand() > density:
                b[i,j] = 0
                b[j,i] = 0
    # Generate a symmetric matrix
    b_symm = (b + b.T) / 2

    G = nx.Graph()
    # For each row in the matrix
    for i in range(n):
        # For each column in the matrix
        for j in range(n):
            # If the edge weight is not 0 add an edge into the graph
            if b_symm[i,j] != 0:
                G.add_edge(i+1, j+1, weight=b_symm[i,j])

    return G


def dijkstra(source, destination, G):
    # Create set P containing only the source node
    P = set(); P.add(source)
    # Create set S containing all other nodes
    S = set(); S.update(G.nodes()); S.remove(source)

    SPE = []
    heap = []

    # If the node is in the neighborhood of the source
    for u in G.neighbors(source):
        # Push the node to the heap with the edge weight from the Graph
        heappush(heap, (G.get_edge_data(source, u)['weight'], u, source))

    # While a shortest path has not been found
    finished = False
    while finished != True:
        # Select node in the heap with minimum distance from P
        u = heappop(heap)

        # Add that node to P
        P.add(u[1])
        # Remove that node from S
        S.remove(u[1])

        # Remove all other paths to that node from the heaplist
        heap = [i for i in heap if i[1] != u[1]]

        # Append the edge to the shortest path edge list
        SPE.append((u[2], u[1]))

        N = [n for n in G.neighbors(u[1]) if n not in P]

        # For each remaining node in S
        for v in N:
            heappush(heap, (G.get_edge_data(u[1], v)['weight'] + u[0], v,
                            u[1]))

        # If S is empty of the accepted node is the destination node,
        # exit the loop
        if len(S) == 0 or u[1] == destination:
            finished = True

    # Clean up edge-list, remove redundant edges that are not on the
    # shortest path. This step is only necessary for generating the shortest
    # path instead of a shortest path spanning tree.
    for i in range(len(SPE) - 1, 0, -1):
        if SPE[i][0] != SPE[i - 1][1]:
            SPE.remove(SPE[i - 1])

    SPE = [i[0] for i in SPE]
    SPE.append(destination)

    return SPE


def bi_dijkstra(source, destination, G):
    # Create distance dictionary P containing only the source node
    P = {source: 0}
    # Create set S containing all other nodes
    S = set(); S.update(G.nodes()); S.remove(source); S.remove(destination)
    # Create distance dictionary T containing only the destination node
    T = {destination: 0}

    # Create the forward search heap
    f_heap = []
    # Create the reverse search heap
    r_heap = []

    # Create the forward path edge list
    FSPE = []
    # Create the reverse path edge list
    RSPE = []

    # Add nodes in the neighbourhood of the source node to the forward
    # search heap
    for node in G.neighbors(source):
        heappush(f_heap, (G.get_edge_data(source, node)['weight'], node,
                        source))

    # Add nodes in the neighbourhood of the destination node to the reverse
    # search heap
    for node in G.neighbors(destination):
        heappush(r_heap, (G.get_edge_data(destination, node)['weight'], node,
                        destination))

    # While a connecting node has not been found
    finished = False
    while finished != True:
        # Select node in forward heap with minimum distance from source
        u_f = heappop(f_heap)

        # Add distance to that node to the forward distance dictionary
        P[u_f[1]] = u_f[0]

        # Calculate neighbourhood of new node
        N = [n for n in G.neighbors(u_f[1]) if n not in P.keys()]

        # Try to remove node from S
        try:
            S.remove(u_f[1])
        except:
            # If it cannot be removed, it means this node has already been
            # scanned. Stop the search.
            W = u_f[1]
            finished = True

        # Remove all other paths to that node from the heaplist
        f_heap = [i for i in f_heap if i[1] != u_f[1]]

        # Append node to forward path edgelist
        FSPE.append((u_f[2], u_f[1]))

        # Attach neighbourhood nodes to forward search heap
        for v in N:
            heappush(f_heap, (G.get_edge_data(u_f[1], v)['weight'] + u_f[0], v,
                            u_f[1]))

        # Select node in forward heap with minimum distance from destination
        u_r = heappop(r_heap)

        # Add distance to that node to the reverse distance dictionary
        T[u_r[1]] = u_r[0]

        # Calculate neighbourhood of new node
        N = [n for n in G.neighbors(u_r[1]) if n not in T.keys()]

        # Try to remove node from S
        try:
            S.remove(u_r[1])
        except:
            # If it cannot be removed, it means this node has already been
            # scanned. Stop the search.
            W = u_r[1]
            finished = True

        # Remove all other paths to that node from the heaplist
        r_heap = [i for i in r_heap if i[1] != u_r[1]]

        # Append node to forward path edgelist
        RSPE.append((u_r[2], u_r[1]))

        # Attach neighbourhood nodes to forward search heap
        for v in N:
            heappush(r_heap, (G.get_edge_data(u_r[1], v)['weight'] + u_r[0], v,
                              u_r[1]))

    # Once the connecting node has been found, the search is stopped and the
    # path length through this node is calculated
    current_shortest_path = T[W] + P[W]

    # For all the remaining forward path neighbourhood nodes, add these to
    # the forward path edgelist and shortest path spanning tree
    for i in range(len(f_heap)):
        u = heappop(f_heap)
        if u[1] not in P.keys():
            P[u[1]] = u[0]
            FSPE.append((u[2], u[1]))

    # For all the remaining reverse path neighbourhood nodes, add these to
    # the reverse path edgelist and shortest path spanning tree
    for i in range(len(r_heap)):
        u = heappop(r_heap)
        if u[1] not in T.keys():
            P[u[1]] = u[0]
            FSPE.append((u[2], u[1]))

    # Find set of nodes that exist in both spanning trees
    connecting_set = [i for i in P.keys() if i in T.keys()]

    # For each of these nodes, if the total distance is shorter than the
    # current shortest path, set this node as the connecting node
    for i in connecting_set:
        shortest_path = T[i] + P[i]
        if shortest_path < current_shortest_path:
            current_shortest_path = shortest_path
            W = i

    # Clean up and combine edge-lists, remove redundant edges that are not on
    # the shortest path. This step is only necessary for generating the
    # shortest path instead of a shortest path spanning tree.

    for i in range(len(FSPE) - 1, 0, -1):
        if FSPE[i][1] == W:
            break
        else:
            FSPE.remove(FSPE[i])

    for i in range(len(FSPE) - 1, 0, -1):
        if FSPE[i][0] != FSPE[i - 1][1]:
            FSPE.remove(FSPE[i - 1])

    for i in range(len(RSPE) - 1, 0, -1):
        if RSPE[i][1] == W:
            break
        else:
            RSPE.remove(RSPE[i])

    FSPE.append((RSPE[-1][1], RSPE[-1][0]))

    for i in range(len(RSPE) - 1, 0, -1):
        if RSPE[i][0] != RSPE[i - 1][1]:
            RSPE.remove(RSPE[i - 1])
        else:
            FSPE.append((RSPE[i][1], RSPE[i][0]))

    FSPE = [i[0] for i in FSPE]
    FSPE.append(destination)

    return FSPE


def main():
    # Set seed
    np.random.seed(124)
    # Create random positive graph
    max_node = 1000
    R = rand_pos_graph(max_node, 0.05)

    t0 = time()
    print(dijkstra(1, max_node, R))
    t1 = time()
    print(t1 - t0)
    print(bi_dijkstra(1, max_node, R))
    t2 = time()
    print(t2 - t1)
    print(nx.shortest_path(R, source=1, target=max_node, weight='weight'))
    t3 = time()
    print(t3 - t2)

if __name__ == "__main__":
    main()