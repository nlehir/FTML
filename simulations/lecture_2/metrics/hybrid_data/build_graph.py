import math
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import pandas as pd
from graphviz import Graph

# Load the data to a pandas dataframe
dataframe = pd.read_csv("csv_files/complex_data.csv")

# get the number of players
nb_players = len(dataframe.index)

# general info on the dataframe
print("---\ngeneral info on the dataframe")
print(dataframe.info())

# print the columns of the dataframe
print("---\ncolumns of the dataset")
print(dataframe.columns)

# print the first 10 lines of the dataframe
print("---\nfirst lines")
print(dataframe.head(10))

# print the correlation matrix of the dataset
print("---\nCorrelation matrix")
print(dataframe.corr())

# print the standard deviation
print("---\nStandard deviation")
print(dataframe.std())

# get specific values in the dataframe
player_id = 1
print("---\nall info on player " + str(player_id))
print(dataframe.loc[player_id])


def compute_dissimilarity(player_1_id, player_2_id):
    """
    Compute  dissimilarity betwwen two players
    based on their id.

    The meal is not a quantitative attribute.
    It is called a categorical variable.
    We must handle it differently than quantitative
    attributes.
    """
    player_1_note = dataframe.loc[player_1_id][1]
    player_2_note = dataframe.loc[player_2_id][1]

    player_1_speed = dataframe.loc[player_1_id][2]
    player_2_speed = dataframe.loc[player_2_id][2]

    player_1_meal = dataframe.loc[player_1_id][3]
    player_2_meal = dataframe.loc[player_2_id][3]

    if player_1_meal == player_2_meal:
        dissimilarity_meal = 0
    else:
        dissimilarity_meal = 15

    # we build a hybrid dissimilarity
    dissimilarity = math.sqrt(
        (player_1_note - player_2_note) ** 2
        + 3 * (player_1_speed - player_2_speed) ** 2
        + dissimilarity_meal
    )

    print("----")
    player_1_name = dataframe.loc[player_1_id]["Name"]
    player_2_name = dataframe.loc[player_2_id]["Name"]
    print(
        f"plyr 1 {player_1_name}, plyr 2 {player_2_name}, dissimilarity: {dissimilarity}"
    )
    return dissimilarity


# build a dissimilarity matrix
dissimilarity_matrix = np.zeros((nb_players, nb_players))
print("compute dissimilarities")
for player_1_id in range(nb_players):
    for player_2_id in range(nb_players):
        dissimilarity = compute_dissimilarity(player_1_id, player_2_id)
        dissimilarity_matrix[player_1_id, player_2_id] = dissimilarity

print("dissimilarity matrix")
print(dissimilarity_matrix)

threshold = 15
# build a graph from the dissimilarity
dot = Graph(comment="Graph created from complex data", strict=True)
for player_id in range(nb_players):
    player_name = dataframe.loc[player_id][0]
    dot.node(player_name)

for player_1_id in range(nb_players):
    # we use an undirected graph so we do not need
    # to take the potential reciprocal edge
    # into account
    for player_2_id in range(nb_players):
        # no self loops
        if not player_1_id == player_2_id:
            player_1_name = dataframe.loc[player_1_id][0]
            player_2_name = dataframe.loc[player_2_id][0]
            # use the threshold condition
            # EDIT THIS LINE
            if dissimilarity_matrix[player_1_id, player_2_id] > threshold:
                dot.edge(
                    player_1_name,
                    player_2_name,
                    color="darkolivegreen4",
                    penwidth="1.1",
                )

# visualize the graph
dot.attr(label=f"threshold {threshold}", fontsize="20")
graph_name = f"images/complex_data_threshold_{threshold}"
dot.render(graph_name)
