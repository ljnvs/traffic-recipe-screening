"""Typed legality rules for a symbolic traffic component space, not a model factory."""


def violations(c, capabilities):
    errors = []
    a, s, g = c["assembly"], c["spatial_operator"], c["graph_source"]
    p, decoder, encoding = c["parameterization"], c["decoder"], c["node_encoding"]
    if s in ("none", "dense_attention") and g != "none":
        errors.append("non-graph operator must not leave an unused graph source")
    if s in ("powers_graph_conv", "chebyshev_graph_conv") and g in ("none", "geo_sem_masks"):
        errors.append("graph convolution requires a compatible support")
    if s == "powers_graph_conv" and g == "provided_laplacian":
        errors.append("scaled Laplacian uses the declared Chebyshev operator")
    if s == "geo_sem_attention" and g != "geo_sem_masks":
        errors.append("heterogeneous attention requires both geographic and semantic masks")
    if g == "geo_sem_masks" and s != "geo_sem_attention":
        errors.append("mask artifact is not a polynomial graph-convolution support")
    if p == "node_pool" and (a not in ("graph_recurrent", "graph_sandwich", "dilated_graph_stack") or s not in ("powers_graph_conv", "chebyshev_graph_conv")):
        errors.append("NAPL requires an explicit graph-feature kernel and matching basis size")
    if decoder == "autoregressive" and a != "graph_recurrent":
        errors.append("initial autoregressive decoder requires a recurrent-state contract")
    if c["delay_pattern"] == "on" and a != "heterogeneous_attention":
        errors.append("delay key update requires the geographic-pattern branch")
    if a == "node_mlp" and (s != "none" or g != "none" or p != "shared" or decoder != "direct"):
        errors.append("node MLP is an independent-per-node summary path")
    if a == "graph_sandwich" and (s not in ("none", "chebyshev_graph_conv") or g not in ("none", "provided_laplacian") or decoder != "direct"):
        errors.append("sandwich block requires a validated spectral support or no-propagation ablation")
    if a == "dilated_graph_stack" and s not in ("none", "powers_graph_conv", "chebyshev_graph_conv"):
        errors.append("dilated stack graph-feature slot cannot accept unimplemented attention path")
    if a == "graph_recurrent" and s not in ("none", "powers_graph_conv", "chebyshev_graph_conv"):
        errors.append("recurrent graph gates require graph-feature or self-only transforms")
    if a == "axis_attention" and (s not in ("none", "dense_attention") or p != "shared" or decoder != "direct"):
        errors.append("axis Transformer uses node attention or its ablation with direct output")
    if a == "heterogeneous_attention" and (s != "geo_sem_attention" or p != "shared" or decoder != "direct" or encoding not in ("none", "laplacian_position")):
        errors.append("initial PD-style coupled heads require the declared heterogeneous contract")
    if encoding == "laplacian_position" and not capabilities["laplacian_positions"]:
        errors.append("Laplacian positional artifact has not been validated")
    if g in ("provided_walk", "provided_laplacian", "provided_plus_adaptive", "geo_sem_masks") and not capabilities["provided_graph"]:
        errors.append("indexed provided graph unavailable")
    if g == "provided_laplacian" and not capabilities["symmetric_graph"]:
        errors.append("symmetric spectral graph unavailable")
    if g == "geo_sem_masks" and not capabilities["training_semantic_graph"]:
        errors.append("train-only semantic graph unavailable")
    if c["delay_pattern"] == "on" and not capabilities["training_pattern_prototypes"]:
        errors.append("train-only history prototypes unavailable")
    if c["calendar"] == "verified_day_week" and not capabilities["verified_calendar"]:
        errors.append("verified actual calendar unavailable")
    if c["calendar"] == "relative_daily_slot" and not capabilities["relative_daily_slot"]:
        errors.append("relative daily phase unavailable")
    return errors


def is_valid(c, capabilities):
    return not violations(c, capabilities)
