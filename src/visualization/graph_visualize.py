import networkx as nx
import matplotlib.patches as mpatches


def _networ_legend(ax):
    legend_dict = {
        'Nodes': '#65A754',
        'Target': '#0E0E0E',
        'User Input': '#8E539F',
        'Blue Team': '#4A7DB3',
        'Red Team': '#D2352B'
    }

    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key],
                                  label=key,
                                  capstyle='round')
        patchList.append(data_key)
    ax.legend(handles=patchList,
              bbox_to_anchor=(0, 1),
              loc='lower left',
              fontsize=16,
              ncol=len(legend_dict.keys()))
    return None


def draw_graph(G, ax):
    pos = nx.get_node_attributes(G, 'position')
    weights = list(nx.get_node_attributes(G, 'node_weight').values())
    node_color = list(nx.get_node_attributes(G, 'node_color').values())

    # Draw the graph
    nx.draw(G,
            pos,
            node_size=weights,
            node_color=node_color,
            ax=ax,
            edgecolors='k')
    _networ_legend(ax)
    return None
