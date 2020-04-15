from functions import get_filename_list
import json
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

if not "citation_graph.p" in os.listdir("."):
    citation_graph = {}

    files = get_filename_list()

    for k, file in enumerate(files):
        print(k)
        contents = json.loads(open(file).read())
        citation_graph[contents["metadata"]["title"]] = [x["title"] for x in contents["bib_entries"].values()]

    pickle.dump(citation_graph, open("citation_graph.p", "wb"))

else:
    citation_graph = pickle.load(open("citation_graph.p", "rb"))


G = nx.Graph()

count = 0
reference_count = {}

papers = citation_graph.keys()

ids = {}

with open("citation_graph.dot", "w") as dot_file:

    dot_file.write("strict digraph { \n")
    for paper, citations in citation_graph.items():

        ids[paper] = count

        print(count)
        count += 1

        if not paper in reference_count.keys():
            reference_count[paper] = 0

        for citation in citations:

            if citation in papers:

                if not citation in reference_count.keys():
                    reference_count[citation] = 0
                    count += 1
                    ids[citation] = count
                reference_count[citation] += 1

                G.add_edge(paper, citation)
                dot_file.write(f"  {ids[paper]} -> {ids[citation]} \n ")

    dot_file.write("} \n")


pr = nx.pagerank(G, alpha=0.9)

print("ok")

# node_sizes = []
# for node in G.nodes:
#     node_sizes.append(reference_count[node])
#
# with open("sizes.dot", "w") as sizes:
#     for k, size in ids.items():
#

# print("Drawing graph")
# nx.draw_networkx(G, with_labels=False, node_size=node_sizes)
# plt.show()
