## reader.py
## Read RST parsing results
## Author: Yangfeng Ji
## Date: 08-28-2016
## Time-stamp: <yangfeng 12/08/2016 21:41:17>

from bracket_reader import BracketReader

class RSTReader(object):
    def __init__(self, fmerge, fbracket, lowercase=True):
        self.fbracket = fbracket
        self.fmerge = fmerge
        self.textdepths = None
        self.textrelas = None
        self.segtexts = None
        self.pnodes = None
        self.lowercase = lowercase
        self.nn_rels = set()
        self.ns_rels = set()
        self.sn_rels = set()


    def _load_segmentation(self):
        """ Load discourse segmentation results
        """
        segs = {}
        with open(self.fmerge, 'r') as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split("\t")
                token, idx = None, int(items[-1])
                if self.lowercase:
                    token = items[0].lower()
                else:
                    token = items[0]
                try:
                    segs[idx] += (" " + token)
                except KeyError:
                    segs[idx] = token
        self.segtexts = segs


    def _load_brackets(self):
        """ Load bracketing results and convert it 
            to a dependency structure
        """
        reader = BracketReader()
        reader.read(self.fbracket)
        deps = reader.convert()
        # print deps
        depths, relas, pnodes = {}, {}, {}
        for dep in deps:
            relas[dep[1]] = dep[2]
            if dep[0] == 'ROOT':
                depths[dep[1]] = 1
            else:
                depths[dep[1]] = depths[dep[0]] + 1
            if dep[0] == 'ROOT':
                pnodes[dep[1]] = 0
            else:
                pnodes[dep[1]] = dep[0]
        self.textdepths = depths
        self.textrelas = relas
        self.pnodes = pnodes
        self.nn_rels = reader.nn_rels
        self.ns_rels = reader.nn_rels
        self.sn_rels = reader.sn_rels


    def read(self):
        self._load_segmentation()
        self._load_brackets()
        return self.segtexts, self.textdepths, self.textrelas


def test():
    fmerge = "/Users/elisa/Documents/CompLing/discourse/parsers/fengHirst_RSTParser/texts/results/train-221022.txt.merge"
    fbracket = "/Users/elisa/Documents/CompLing/discourse/parsers/fengHirst_RSTParser/texts/results/train-221022.txt.brackets"
    rstreader = RSTReader(fmerge, fbracket)
    rstreader.read()
    print "Depths: ", rstreader.textdepths
    print "Texts: ", rstreader.segtexts
    print "Relations: ", rstreader.textrelas
    print "Pnodes: ", rstreader.pnodes


if __name__ == '__main__':
    test()
