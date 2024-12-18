import re
import graph_tool as gt
import numpy as np

class Graph(object):

    def __init__(self, path=None, G=None, name=None, abbr=None):
        if path is not None:
            if path.endswith(".csv"):
                self.G = gt.load_graph_from_csv(path)
            else:
                self.G = gt.load_graph(path)
        else:
            self.G = gt.Graph(G)
        self.name = name
        self.abbr = abbr

    def n(self):
        return self.G.num_vertices()

    def m(self):
        return self.G.num_edges()

    def degrees(self):
        return self.G.get_out_degrees(self.G.get_vertices())

    def max_degree(self):
        return np.amax(self.degrees())

    def edgesba(self):        
        return self.vp.edgesba.a

    def I(self):
        return self.vp.I.a
    
    def T(self):
        return self.vp.T.a
    
    def ed(self):
        return self.vp.ed.a
    
    def ebc(self):
        return self.vp.ebc.a
    
    def ps(self):
        return self.vp.ps.a    

    def __getattr__(self, *args, **kwargs):
        return getattr(self.G, *args, **kwargs)