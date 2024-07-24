

pp = {
        'Al_sg15_fr.upf': ['Al', 'sg15', 'fr']
        'Fe_sg15_sr.upf': ['Fe', 'sg15', 'sr']
        }

class TagSearch:

    def __init__(self, pp):
        self.inv_pp = invert(pp)


    def __call__(self, *args):
        inv_pp

    def invert(pp):
        res = {}
        for key, value in pp.items():
            for v in value:
                if v not in res:
                    res[v] = []
                res[v].append(key)
    
        # convert list to set TBD
    
        return res



