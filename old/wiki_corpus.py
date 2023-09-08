from gensim.corpora import WikiCorpus, MmCorpus
from gensim.test.utils import datapath, get_tmpfile
import gensim.downloader as api

def prep_wiki_corpus():
    # Saved to: C:\Users\paulc\Anaconda3\envs\py311\Lib\site-packages\gensim\test\test_data
    fp_corpus = "enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2"
    path_corpus = datapath(fp_corpus)
    wiki = WikiCorpus(path_corpus)
    fp_serialize = "wiki-corpus.mm"
    path_serialize = get_tmpfile(fp_serialize)
    MmCorpus.serialize(path_serialize, wiki)



# https://radimrehurek.com/gensim/wiki.html
# https://radimrehurek.com/gensim/scripts/make_wikicorpus.html

if __name__ == '__main__':
    # prep_wiki_corpus()
    load_wiki()
