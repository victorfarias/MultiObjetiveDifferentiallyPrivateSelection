from dp_mechanisms.graph.graph import Graph
from dp_mechanisms.graph.ego_influence import compute_edgesba

def precompute_egc(source, target):
    G = Graph(source)
    compute_I_ego_betweenness(G)
    G.save(target)

def precompute_edgesba(source, target):
    G = Graph(source)
    compute_edgesba(G)
    G.save(target)

if __name__ == "__main__":
    # precompute_edgesba("./data/enron/enron_ed_ebc.graphml", "./data/enron/enron_ed_ebc_edgesba.graphml")
    precompute_edgesba("./data/github/github_ed_ebc.graphml", "./data/github/github_ed_ebc_edgesba.graphml")
    # precompute_egc("./data/dblp/com-dblp.ungraph.graphml", "./data/dblp/com-dblp.ungraph_egc.graphml")
    
