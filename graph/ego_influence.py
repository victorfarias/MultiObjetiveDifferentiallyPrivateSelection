import graph_tool as gt
import numpy as np

from graph_tool.centrality import betweenness

############ Ego Betweness ############
def compute_ebc_node(G, v):
    edges_v = G.get_out_edges(v)
    neighbors_v = edges_v[:,1]
    mask = np.full((G.n(),), False)
    mask[neighbors_v] = True
    mask_p = G.new_vertex_property("bool")
    mask_p.a = mask
    mask_p[v] = True
    Gv = gt.GraphView(G, vfilt=mask_p)
    vp, _ = betweenness(Gv, norm=False)
    G.vp.ebc[v] = vp[v]
    return G.vp.ebc.a

def compute_ebc(G):
    G.vp.ebc = G.new_vertex_property("double")    
    n = G.n()
    for v_id in G.get_vertices():     
        print(v_id, n)
        compute_ebc_node(G, G.vertex(v_id))

def change_to_ebc(G):
    G.vp.I = G.vp.ebc

############ Ego Density ############
def compute_ed_node(G, v):
    degree = v.out_degree()
    if degree == 1:
        G.vp.ed[v] = 1.
        G.vp.T[v] = 1
        return         
    edges_v = G.get_out_edges(v)
    neighbors_v = edges_v[:,1]
    mask = np.full((G.n(),), False)
    mask[neighbors_v] = True
    mask_p = G.new_vertex_property("bool")
    mask_p.a = mask
    mask_p[v] = True
    Gv = gt.GraphView(G, vfilt=mask_p)
    num_edges = Gv.num_edges()    
    T = num_edges - degree
    density = T/((degree)*(degree-1))
    G.vp.ed[v] = density    
    G.vp.T[v] = T
    

def compute_ed(G):
    G.vp.ed = G.new_vertex_property("double")    
    G.vp.T = G.new_vertex_property("long")
    n = G.n()
    for v_id in G.get_vertices():     
        print(v_id, n)
        compute_ed_node(G, G.vertex(v_id))
    return G.vp.ed.a

def change_to_ed(G):
    G.vp.I = G.vp.ed
    
######## EBCD #########
def change_to_ebcd(G):
    G.vp.I = G.new_vertex_property("double")    
    G.vp.I.a = G.ed()*G.ebc()

###### Edges between alters #########

def compute_edgesba_node(G, v):
    edges_v = G.get_out_edges(v)
    neighbors_v = edges_v[:,1]
    mask = np.full((G.n(),), False)
    mask[neighbors_v] = True
    mask_p = G.new_vertex_property("bool")
    mask_p.a = mask
    mask_p[v] = True
    Gv = gt.GraphView(G, vfilt=mask_p)
    num_edges = Gv.num_edges()
    degree = len(neighbors_v)
    G.vp.edgesba[v] = num_edges - degree

def compute_edgesba(G):
    G.vp.edgesba = G.new_vertex_property("double")    
    n = G.n()
    for v_id in G.get_vertices():     
        print(v_id, n)
        compute_edgesba_node(G, G.vertex(v_id))
    return G.vp.edgesba.a