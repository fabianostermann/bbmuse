import logging

logger = logging.getLogger(__name__)

def plot_dependency_graph(project, filename="graph.html", include_uses=True):
    """Plot the bipartite dependency graph (Representations & Modules) layered by hard deps using networkx and pyvis."""
    if not project.controller:
        raise RuntimeError("Controller has not been built yet. Call build_all() first.")

    try:
        import networkx as nx
        from pyvis.network import Network
    except Exception:
        logger.exception("Plotting of dependency graph failed. networkx or pyvis package is missing.")
        return

    def mod_id(handler):
        name = handler.get_name() if hasattr(handler, "get_name") else str(handler)
        return f"mod:{name}"

    def repr_id(name):
        return f"repr:{name}"

    # --- Build bipartite directed graph ---
    G = nx.DiGraph()
    module_nodes = []
    repr_nodes = set()
    requires_edges = []
    provides_edges = []
    uses_edges = []

    for m in project.module_handlers:
        mid = mod_id(m)
        if mid not in G:
            G.add_node(mid, kind="module")
            module_nodes.append(mid)

        for r in m.get_requires():
            rid = repr_id(r)
            repr_nodes.add(rid)
            requires_edges.append((rid, mid))

        for p in m.get_provides():
            pid = repr_id(p)
            repr_nodes.add(pid)
            provides_edges.append((mid, pid))

        if include_uses and hasattr(m, "get_uses"):
            for u in (m.get_uses() or []):
                uid = repr_id(u)
                repr_nodes.add(uid)
                uses_edges.append((uid, mid))

    for rid in repr_nodes:
        if rid not in G:
            G.add_node(rid, kind="representation")

    G.add_edges_from(requires_edges, kind="requires")
    G.add_edges_from(provides_edges, kind="provides")
    if include_uses:
        G.add_edges_from(uses_edges, kind="uses")

    # --- Compute dependency layers using only hard edges ---
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

    for n in G.nodes():
        if n not in layers:
            layers[n] = 0

    for n, l in layers.items():
        G.nodes[n]["subset"] = l
    pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal", scale=2.0)

    # --- Build pyvis network ---
    # use toggle_physics() rather than net.options.physics.enabled = False
    net = Network(directed=True, height="750px", width="100%")
    net.toggle_physics(True)

    labels = {n: n.split(":", 1)[1] for n in G.nodes()}

    # add the nodes directly instead of from_nx() plus later modification, so
    # the attributes are set for certain and not overwritten by from_nx
    for node_id in G.nodes():
        kind = G.nodes[node_id]["kind"]
        x, y = pos[node_id]
        net.add_node(
            node_id,
            label=labels[node_id],
            title=node_id,
            x=float(x * 500),
            y=float(y * 500),
            physics=False,
            color="#FFEDB5" if kind == "representation" else "#CDEAFE",
            shape="dot"     if kind == "representation" else "box",
            size=25         if kind == "representation" else 30,
            font={"size": 14},
        )

    # dashes takes a boolean; pyvis does not accept an array such as [5, 5]
    for u, v, data in G.edges(data=True):
        kind = data.get("kind", "unknown")
        if kind == "requires":
            net.add_edge(u, v, color="black",   width=2,   dashes=False, title="requires")
        elif kind == "provides":
            net.add_edge(u, v, color="#666666", width=1.5, dashes=True,  title="provides")
        elif kind == "uses":
            net.add_edge(u, v, color="#CCCCCC", width=0.8, dashes=True,  title="uses")
        else:
            net.add_edge(u, v)

    # path taken from project.config; write_html() rather than show(), which in
    # newer versions needs notebook=False and opens a browser window
    if filename is None:
        # no file path given -> open in the browser (the analogue of plt.show())
        net.show("graph.html", notebook=False)
    else:
        output_path = str(project.config.get_project_dir().joinpath(filename))
        net.write_html(output_path)
        logger.info("Dependency graph saved to %s", output_path)
