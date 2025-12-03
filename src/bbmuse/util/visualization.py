import logging

logger = logging.getLogger(__name__)

# TODO: include groups in visualization!
# TODO: Try to repeat visualization multiple times (iterate over seed) and count "crossed lines"
def plot_dependency_graph(project, filename="graph.png", include_uses=True):
    """Plot the bipartite dependency graph (Representations & Modules) layered by hard deps.
    
    Edges:
    - requires: representation -> module (solid)
    - provides: module -> representation (dashed)
    - uses:     representation -> module (dotted, optional, does not affect layering)
    """
    if not project.controller:
        raise RuntimeError("Controller has not been built yet. Call build_all() first.")

    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        logger.exception("Plotting of dependency graph failed. Matplotlib or networkx package is missing.")
        return

    # --- Helpers to build stable ids and readable labels ---
    def mod_id(handler):
        name = handler.get_name() if hasattr(handler, "get_name") else str(handler)
        return f"mod:{name}"

    def repr_id(name):
        return f"repr:{name}"

    # --- Build bipartite directed graph ---
    G = nx.DiGraph()
    module_nodes = []
    repr_nodes = set()
    requires_edges = []   # repr -> mod (hard)
    provides_edges = []   # mod -> repr (hard)
    uses_edges = []       # repr -> mod (soft, optional)

    for m in project.module_handlers:
        mid = mod_id(m)
        if mid not in G:
            G.add_node(mid, kind="module")
            module_nodes.append(mid)

        # requires: representation -> module (hard)
        for r in m.get_requires():
            rid = repr_id(r)
            repr_nodes.add(rid)
            requires_edges.append((rid, mid))

        # provides: module -> representation (hard)
        for p in m.get_provides():
            pid = repr_id(p)
            repr_nodes.add(pid)
            provides_edges.append((mid, pid))

        # uses: representation -> module (soft, optional)
        if include_uses and hasattr(m, "get_uses"):
            for u in (m.get_uses() or []):
                uid = repr_id(u)
                repr_nodes.add(uid)
                uses_edges.append((uid, mid))

    # Ensure all representation nodes exist
    for rid in repr_nodes:
        if rid not in G:
            G.add_node(rid, kind="representation")

    # Add edges with metadata (all visible in the plot)
    G.add_edges_from(requires_edges, kind="requires")
    G.add_edges_from(provides_edges, kind="provides")
    if include_uses:
        G.add_edges_from(uses_edges, kind="uses")

    # --- Compute dependency layers (topological levels) using only HARD edges ---
    # We build a temporary graph H with only requires/provides to derive layers.
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(requires_edges)
    H.add_edges_from(provides_edges)

    indeg = {n: 0 for n in H.nodes()}
    for u, v in H.edges():
        indeg[v] += 1

    layers = {}
    current_layer = [n for n, d in indeg.items() if d == 0]
    layer_idx = 0
    while current_layer:
        next_layer = []
        for n in current_layer:
            layers[n] = layer_idx
            for succ in H.successors(n):
                indeg[succ] -= 1
                if indeg[succ] == 0:
                    next_layer.append(succ)
        current_layer = next_layer
        layer_idx += 1

    # Fallback for any unassigned nodes (should not happen without cycles)
    for n in G.nodes():
        if n not in layers:
            layers[n] = 0

    # Attach as node attribute and compute left→right layout by layer
    for n, l in layers.items():
        G.nodes[n]["subset"] = l
    pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal", scale=2.0)

    # --- Draw: nodes (cleaner defaults, no extra edgecolor/linewidth/size args) ---
    labels = {n: n.split(":", 1)[1] for n in G.nodes()}
    repr_nodes_list = list(repr_nodes)
    module_nodes_list = module_nodes

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=repr_nodes_list,
        node_shape="o",
        node_color="#FFEDB5",
        label="Representations",
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=module_nodes_list,
        node_shape="s",
        node_color="#CDEAFE",
        label="Modules",
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    # --- Draw: edges ---
    # requires: solid, slightly transparent so other styles can stand out
    nx.draw_networkx_edges(
        G, pos,
        edgelist=requires_edges,
        arrows=True,
        #arrowstyle="-|>",
        width=1.4,
        alpha=0.9,
        min_source_margin=12,  # <-- shorten at start by 10 pt
        min_target_margin=12,  # <-- shorten at end by 10 pt
    )
    # provides: dashed
    nx.draw_networkx_edges(
        G, pos,
        edgelist=provides_edges,
        arrows=True,
        #arrowstyle="-|>",
        #style="dashed",
        width=1.2,
        alpha=0.9,
        min_source_margin=12,  # <-- shorten at start by 10 pt
        min_target_margin=12,  # <-- shorten at end by 10 pt
    )
    # uses: dotted + curved + drawn last with higher zorder so it doesn't hide behind others
    if include_uses and uses_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=uses_edges,
            arrows=True,
            #arrowstyle="-|>",
            #style="dotted",
            width=1.0,
            alpha=0.2,
            connectionstyle="arc3,rad=0.25",  # <-- curve the edge to avoid overlap
            min_source_margin=12,  # <-- shorten at start by 10 pt
            min_target_margin=12,  # <-- shorten at end by 10 pt
        )

    # --- Legend & finalize ---
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#FFEDB5", markeredgecolor="black", label="Representation"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#CDEAFE", markeredgecolor="black", label="Module"),
        Line2D([0], [0], linestyle="-", color="black", label="requires (repr → module)"),
        Line2D([0], [0], linestyle="-", color="black", label="provides (module → repr)"),
    ]
    if include_uses:
        legend_elements.append(Line2D([0], [0], linestyle="-", color="black", alpha=0.2, label="uses (repr → module)"))
    plt.legend(handles=legend_elements, loc="best", fontsize=8)

    plt.title(project.config["application"]["name"])
    plt.tight_layout()
    
    if filename is None:
        plt.show()
    else:
        filename = project.config.get_project_dir().joinpath(filename)
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        logger.info("Dependency graph saved to %s", filename)

    plt.close()

