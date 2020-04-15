import os
from functions import get_filename_list
import json
import networkx as nx
import re
import pickle
import pygraphviz
from networkx.drawing.nx_pydot import write_dot
from copy import deepcopy

import matplotlib.pyplot as plt

def get_authors(json_file):
    return [re.sub("[\s\-\.\~\d\#\(\)\?\/\:\;\$\*\]\[€•†]|(&apos;)|(&amp;)|(&quot;)", "", f"{x['first']} {x['last']}".strip()).lower() for x
     in json_file["metadata"]["authors"]
            if len(x["first"]) > 2 and len(x["last"]) > 2]


def quick_clean(text):
    aux = re.sub('[^0-9a-zA-Z\s]+', '', text).lower()
    return re.sub('\s{2,}', ' ', aux)


if not "papers.p" in os.listdir("."):

    papers = {}
    authors = {}
    citations = {}
    titles = {}

    for k, filename in enumerate(get_filename_list()):

        print(k)

        json_file = json.loads(open(filename).read())
        paper_id = json_file["paper_id"]

        title = quick_clean(json_file["metadata"]["title"])
        titles[title] = paper_id
        citations[paper_id] = [quick_clean(x["title"]) for x in json_file["bib_entries"].values()]

        paper_authors = set(get_authors(json_file))

        papers[paper_id] = paper_authors

        for a in paper_authors:
            if not a in authors.keys():
                authors[a] = []

            authors[a].append(paper_id)

    pickle.dump(papers, open('papers.p', 'wb'))
    pickle.dump(authors, open('author.p', 'wb'))
    pickle.dump(citations, open('citations.p', 'wb'))
    pickle.dump(titles, open('titles.p', 'wb'))
else:

    papers = pickle.load(open('papers.p', 'rb'))
    authors = pickle.load(open('author.p', 'rb'))
    citations = pickle.load(open('citations.p', 'rb'))
    titles = pickle.load(open('titles.p', 'rb'))

################################################################################

if 'citation_paperid.p' not in os.listdir("."):
    citation_paperid = {}  # paper_id: list[paper_id]

    for paper_id, title_list in citations.items():
        citation_paperid[paper_id] = [titles[title] for title in title_list if title in titles.keys()]

    pickle.dump(citation_paperid, open('citation_paperid.p', 'wb'))

else:

    citation_paperid = pickle.load(open('citation_paperid.p', 'rb'))

################################################################################


if 'author_citations.p' not in os.listdir("."):
    author_citations = {}  # author: list[author]

    for paper_id, author_list in papers.items():

        citations = citation_paperid[paper_id]
        cited_authors = set()

        for citation in citations:
            authors = papers[citation]
            cited_authors.update(authors)

        for author in author_list:
            if author not in author_citations.keys():
                author_citations[author] = set()

            author_citations[author].update(cited_authors)

    pickle.dump(author_citations, open('author_citations.p', 'wb'))

else:
    author_citations = pickle.load(open('author_citations.p', 'rb'))

################################################################################

if '07-graph.p' not in os.listdir("."):
    G = nx.DiGraph()

    for author, cited_authors in author_citations.items():

        for other_author in cited_authors:
            G.add_edge(author, other_author)

    pickle.dump(G, open("07-graph.p", 'wb'))
else:
    G = pickle.load(open("07-graph.p", 'rb'))

if 'pr.p' not in os.listdir('.'):
    pr = nx.pagerank(G, alpha=0.9)
    pickle.dump(pr, open("pr.p", 'wb'))
else:
    pr = pickle.load(open("pr.p", 'rb'))


##############################################################################

# removing nodes with less than 10 in or out edges or more than 20, in order to keep the graph small enough
# so it can be drawn in feasible time without much work

small_G = deepcopy(G)

to_delete = [node for node in G.nodes
             if len(small_G.out_edges(node)) > 20
             or len(G.in_edges(node)) > 20
             or len(G.in_edges(node)) < 10
             or len(G.out_edges(node)) < 10]

small_G.remove_nodes_from(to_delete)

Gcc = sorted(nx.connected_components(small_G), key=len, reverse=True)[0]

write_dot(small_G, "small_g.dot")
write_dot(Gcc, "giant_connected_graph.dot")

# remove notes without any in or out edges

with open("small_g.dot", "r", encoding="utf-8") as dotfile:

    dotlines = dotfile.readlines()[5:]

    nodes = dotlines[:1741]
    edges = dotlines[1741:]

    remove = []
    for node in nodes:
        for edge in edges:
            if node in edge:
                break
        else:
            remove.append(node)

    for n in remove:
        nodes.remove(n)

    with open("small_g2.dot", "w", encoding="utf-8") as dotfile2:

        dotfile2.writelines(nodes)
        dotfile2.writelines(edges)
