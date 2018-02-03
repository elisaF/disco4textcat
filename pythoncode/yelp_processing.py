from rst_reader import RSTReader
from os import walk
from os.path import join, basename
from collections import defaultdict
from operator import itemgetter
from string import punctuation
import io, gzip, os, cPickle, sys
import fnmatch


class Doc(object):
    def __init__(self, fname, edus, relas, pnodes, label=None):
        self.fname = fname
        self.edus = edus
        self.relas = relas
        self.pnodes = pnodes
        self.label = label

def parse_fname(fname):
    items = (fname.split(".")[0]).split("-")
    if len(items) != 2:
        raise ValueError("Unexpected length of items: {}".format(items))
    setlabel, fidx = items[0], int(items[1])
    return setlabel, fidx

def load_labels(fname):
    with gzip.open(fname, 'r') as fin:
        labels = fin.read().strip().split("\n")
        print "Load {} labels from file: {}".format(len(labels), fname)
    return labels

def check_token(tok):
    # punctuation only
    is_punc = True
    for c in tok:
        is_punc = is_punc and (c in punctuation)
    # is number
    tok = tok.replace(",","")
    is_number = True
    try:
        float(tok)
    except ValueError:
        is_number = False
    # refine token
    tok = tok.replace("-", "")
    return tok, is_punc, is_number

def get_allbracketsfiles(rpath, suffix="*.brackets"):
    bracketsfiles = []
    for root, dirnames, file_names in os.walk(rpath):
        for file_name in fnmatch.filter(file_names, suffix):
            bracketsfiles.append(join(root, file_name))
    print "Read {} files".format(len(bracketsfiles))
    return bracketsfiles


def get_docdict(bracketsfiles, trn_labels, dev_labels, tst_labels, suffix=".brackets"):
    counter = 0
    nn_rels, ns_rels, sn_rels = set(), set(), set()
    trn_docdict, dev_docdict, tst_docdict = {}, {}, {}
    for fbrackets in bracketsfiles:
        # print "Read file: {}".format(fbrackets)
        fmerge = fbrackets.replace("brackets", "merge")
        rstreader = RSTReader(fmerge, fbrackets)
        try:
            rstreader.read()
            nn_rels.update(rstreader.nn_rels)
            ns_rels.update(rstreader.ns_rels)
            sn_rels.update(rstreader.sn_rels)
        except SyntaxError:
            print "Ignore file: ", fmerge
            counter += 1
            continue

        fname = basename(fmerge).replace(".merge","")
        setlabel, fidx = parse_fname(fname)
        if setlabel == "train":
            doc = Doc(fname, rstreader.segtexts, rstreader.textrelas, rstreader.pnodes, int(trn_labels[fidx])-1)  # convert to scale of 0-4 instead of 1-5
            trn_docdict[fname] = doc
        elif setlabel == "dev":
            doc = Doc(fname, rstreader.segtexts, rstreader.textrelas, rstreader.pnodes,
                      int(dev_labels[fidx]) - 1)
            dev_docdict[fname] = doc
        elif setlabel == "test":
            doc = Doc(fname, rstreader.segtexts, rstreader.textrelas, rstreader.pnodes, int(tst_labels[fidx])-1)
            tst_docdict[fname] = doc
    print "Ignore {} files in total".format(counter)
    with open("relations.sets", 'w') as fout:
        fout.write("\nnn: "+ str(nn_rels))
        fout.write("\nns: "+ str(ns_rels))
        fout.write("\nsn: "+ str(sn_rels))
    return trn_docdict, dev_docdict, tst_docdict

def get_vocab(trn_docs, dev_docs, thresh=10000):
    counts = defaultdict(int)
    rela_vocab = {'root':0}
    for (fname, doc) in trn_docs.iteritems():
        for (eidx, edu) in doc.edus.iteritems():
            tokens = edu.strip().split()
            for tok in tokens:
                tok, is_punc, is_number = check_token(tok)
                if is_punc:
                    continue
                if is_number:
                    tok = "NUMBER"
                counts[tok] += 1
        for (eidx, rela) in doc.relas.iteritems():
            try:
                rela_vocab[rela]
            except KeyError:
                rela_vocab[rela] = len(rela_vocab)
    for (fname, doc) in dev_docs.iteritems():
        for (eidx, edu) in doc.edus.iteritems():
            tokens = edu.strip().split()
            for tok in tokens:
                tok, is_punc, is_number = check_token(tok)
                if is_punc:
                    continue
                if is_number:
                    tok = "NUMBER"
                counts[tok] += 1
        for (eidx, rela) in doc.relas.iteritems():
            try:
                rela_vocab[rela]
            except KeyError:
                rela_vocab[rela] = len(rela_vocab)
    print "Size of the raw vocab: ", len(counts)
    # rank with 
    sorted_counts = sorted(counts.items(), key=itemgetter(1), reverse=True)
    word_vocab, N = {}, 0
    for item in sorted_counts:
        word_vocab[item[0]] = N
        N += 1
        if N >= thresh:
            break
    print "After filtering out low-frequency words, vocab size = ", len(word_vocab)
    return word_vocab, rela_vocab


def refine_with_vocab(edu, vocab):
    t_count, u_count = 0.0, 0.0
    tokens = edu.strip().split()
    new_tokens = []
    for tok in tokens:
        tok, is_punc, is_number = check_token(tok)
        if is_punc:
            continue
        t_count += 1.0
        if is_number:
            tok = "NUMBER"
        try:
            vocab[tok]
        except KeyError:
            u_count += 1.0
            tok = "UNK"
        new_tokens.append(tok)
    return " ".join(new_tokens), t_count, u_count


def write_dict(rvocab, fdict):
    with open(fdict, 'wb') as f:
        cPickle.dump(rvocab, f)


def write_docs(docdict, wvocab, rvocab, outfname, is_trnfile=False, dev_docdict=None, dev_outfname=None):
    print "Write docs into file: {}".format(outfname)
    if is_trnfile:
        w2vfname = outfname.replace("txt", "w2v")
        print "Write train tokens into file: {}".format(w2vfname)
        fw2v = open(w2vfname, 'w')
        w2vfname_dev = dev_outfname.replace("txt", "w2v")
        print "Write dev tokens into file: {}".format(w2vfname_dev)
        fw2v_dev = open(w2vfname_dev, 'w')
    total_count, unk_count = 0.0, 0.0
    with open(outfname, 'w') as fout:
        fout.write("EIDX\tPIDX\tRIDX\tEDU\n")
        for (fname, doc) in docdict.iteritems():
            edus = doc.edus
            for eidx in range(len(edus)):
                edu = edus[eidx+1]
                edu, tc, uc = refine_with_vocab(edu, wvocab)
                total_count += tc
                unk_count += uc
                try:
                    ridx = rvocab[doc.relas[eidx+1]]
                except KeyError:
                    ridx = rvocab['elaboration']
                try:
                    pidx = doc.pnodes[eidx+1]-1
                except KeyError:
                    print "KeyError: ", fname, eidx + 1
                    print doc.pnodes
                line = "{}\t{}\t{}\t{}\n".format(eidx, pidx, ridx, edu)
                fout.write(line)
                # fw2v.write("<s> {} </s>\n".format(edu))
                if is_trnfile:
                    fw2v.write("{}\n".format(edu))
            fout.write("=============\t{}\t{}\n".format(fname, doc.label))
    if is_trnfile:
        with open(dev_outfname, 'w') as fout:
            fout.write("EIDX\tPIDX\tRIDX\tEDU\n")
            for (fname, doc) in dev_docdict.iteritems():
                edus = doc.edus
                for eidx in range(len(edus)):
                    edu = edus[eidx + 1]
                    edu, tc, uc = refine_with_vocab(edu, wvocab)
                    total_count += tc
                    unk_count += uc
                    try:
                        ridx = rvocab[doc.relas[eidx + 1]]
                    except KeyError:
                        ridx = rvocab['elaboration']
                    try:
                        pidx = doc.pnodes[eidx + 1] - 1
                    except KeyError:
                        print "KeyError for dev: ", fname, eidx + 1
                        print doc.pnodes
                    line = "{}\t{}\t{}\t{}\n".format(eidx, pidx, ridx, edu)
                    fout.write(line)
                    # fw2v.write("<s> {} </s>\n".format(edu))
                    fw2v_dev.write("{}\n".format(edu))
                fout.write("=============\t{}\t{}\n".format(fname, doc.label))
    # counts
    print "Total tokens: {}; UNK counts: {}; Ratio: {}".format(total_count, unk_count, (unk_count/total_count))
    # write vocab
    if is_trnfile:
        fw2v.close()
        fw2v_dev.close()
        vocabfname = outfname.replace("txt", "vocab")
        with open(vocabfname, "w") as fout:
            for (key, val) in wvocab.iteritems():
                fout.write("{}\n".format(key))
        print "Write vocab into file: {}".format(vocabfname)


def main():
    if len(sys.argv) != 2:
        print "FORMAT: data_dir"
        sys.exit(1)

    data_dir = sys.argv[1]

    # pls change T and FOLDER at the sametime
    T = 10000
    #FOLDER = "fortextclass-10K"
    #SUFFIX = ".brackets25"
    SUFFIX = "*.brackets"
    # load labels
    trn_labels = load_labels(os.path.join(data_dir, "train.labels.gz"))
    dev_labels = load_labels(os.path.join(data_dir, "dev.labels.gz"))
    tst_labels = load_labels(os.path.join(data_dir, "test.labels.gz"))
    # load all files
    rpath = os.path.join(data_dir, "feng_parses/")
    flist = get_allbracketsfiles(rpath, SUFFIX)
    trn_docdict, dev_docdict, tst_docdict = get_docdict(flist, trn_labels, dev_labels, tst_labels, SUFFIX)
    # get vocabs
    wvocab, rvocab = get_vocab(trn_docdict, dev_docdict, thresh=T)
    # write files
    ftrn = os.path.join(data_dir, "output/trn-yelp.txt")
    fdev = os.path.join(data_dir, "output/dev-yelp.txt")
    write_docs(trn_docdict, wvocab, rvocab, ftrn, is_trnfile=True, dev_docdict=dev_docdict, dev_outfname=fdev)
    ftst = os.path.join(data_dir, "output/tst-yelp.txt")
    write_docs(tst_docdict, wvocab, rvocab, ftst)
    infofname = os.path.join(data_dir, "output/info-yelp.txt")
    fdict = os.path.join(data_dir, "output/relations.p")
    write_dict(rvocab, fdict)
    with open(infofname, 'w') as fout:
        fout.write("Size of the training examples: {}\n".format(len(trn_docdict)))
        fout.write("Size of the development examples: {}\n".format(len(dev_docdict)))
        fout.write("Size of the test examples: {}\n".format(len(tst_docdict)))
        fout.write("Size of the word vocab: {}\n".format(len(wvocab)))
        fout.write("Size of the relation vocab: {}\n".format(len(rvocab)))
        fout.write("Relation mapping: {}\n".format(rvocab))



if __name__ == '__main__':
    main()
