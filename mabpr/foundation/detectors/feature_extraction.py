"""feature_extraction.py"""

import argparse
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pandas as pd
import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.util import interval_lifecycle

warnings.filterwarnings("ignore")

@dataclass
class ControlFlowFeatures:

    dfg_frequency: dict = field(default_factory=dict)

    dependency_strength: dict = field(default_factory=dict)

    activity_frequency: dict = field(default_factory=dict)

    variant_frequency: dict = field(default_factory=dict)

    activity_variant_coverage: dict = field(default_factory=dict)

    ordering_consistency: dict = field(default_factory=dict)

    rework_counts: dict = field(default_factory=dict)

@dataclass
class TimeFeatures:

    activity_duration_mean: dict = field(default_factory=dict)

    waiting_time_mean: dict = field(default_factory=dict)

    waiting_to_duration_ratio: dict = field(default_factory=dict)

    throughput_time_per_variant: dict = field(default_factory=dict)

    throughput_time_mean: float = 0.0

    waiting_time_between: dict = field(default_factory=dict)

@dataclass
class ResourceFeatures:

    handover_count_mean: float = 0.0

    handover_count_per_case: dict = field(default_factory=dict)

    resource_continuity_rate: float = 0.0

    resource_utilization: dict = field(default_factory=dict)

    repeated_participation_rate: dict = field(default_factory=dict)

    waiting_after_handover_mean: dict = field(default_factory=dict)

@dataclass
class BatchFeatures:

    batched_activities: dict = field(default_factory=dict)

    batch_size_mean: dict = field(default_factory=dict)

    batch_interval_mean: dict = field(default_factory=dict)

    periodicity_strength: dict = field(default_factory=dict)

@dataclass
class ProcessGraphFeatures:

    graph: Any = None

    node_metadata: dict = field(default_factory=dict)

    activity_to_node: dict = field(default_factory=dict)

@dataclass
class FeatureStore:
    control_flow: ControlFlowFeatures = field(default_factory=ControlFlowFeatures)
    time: TimeFeatures = field(default_factory=TimeFeatures)
    resource: ResourceFeatures = field(default_factory=ResourceFeatures)
    batch: BatchFeatures = field(default_factory=BatchFeatures)
    process_graph: ProcessGraphFeatures = field(default_factory=ProcessGraphFeatures)
    total_cases: int = 0
    activities: list = field(default_factory=list)

def _compute_active_duration_overrides(log) -> dict:
    """For each activity instance (case, concept:name, instance_ordinal),"""

    overrides: dict = {}
    for trace in log:
        case_id = trace.attributes.get("concept:name")
        if case_id is None:
            continue
        instance_counter: dict = {}
        active_states: dict = {}
        for ev in trace:
            act = ev.get("concept:name")
            trans = (ev.get("lifecycle:transition") or "").lower()
            ts = ev.get("time:timestamp")
            if act is None or ts is None:
                continue

            if trans == "start":
                ord_idx = instance_counter.get(act, 0)
                instance_counter[act] = ord_idx + 1
                active_states[act] = {
                    "instance_idx":          ord_idx,
                    "accumulated_s":         0.0,
                    "current_active_start":  ts,
                    "saw_suspend_or_resume": False,
                }
            elif trans == "resume":
                state = active_states.get(act)
                if state is not None and state["current_active_start"] is None:
                    state["current_active_start"] = ts
                    state["saw_suspend_or_resume"] = True
            elif trans == "suspend":
                state = active_states.get(act)
                if state is not None and state["current_active_start"] is not None:
                    state["accumulated_s"] += (ts - state["current_active_start"]).total_seconds()
                    state["current_active_start"] = None
                    state["saw_suspend_or_resume"] = True
            elif trans in ("complete", "ate_abort", "withdraw"):
                state = active_states.pop(act, None)
                if state is not None:
                    if state["current_active_start"] is not None:
                        state["accumulated_s"] += (ts - state["current_active_start"]).total_seconds()

                    if state["saw_suspend_or_resume"]:
                        overrides[(case_id, act, state["instance_idx"])] = state["accumulated_s"]

    return overrides

def prepare_dataframe(log_path: str) -> pd.DataFrame:
    """Reads a XES log and returns a flat DataFrame with a proper"""

    log = pm4py.read_xes(log_path, return_legacy_log_object=True)

    for trace in log:
        for i, event in enumerate(trace):
            trans = event.get("lifecycle:transition", "").upper()
            resource = event.get("org:resource")

            if trans in ("START", "COMPLETE") and (resource is None or str(resource) == "nan"):
                act = event["concept:name"]

                for j in range(i - 1, -1, -1):
                    prev = trace[j]
                    if (prev["concept:name"] == act
                        and prev.get("lifecycle:transition", "").upper() == "SCHEDULE"):

                        prev_resource = prev.get("org:resource")
                        if prev_resource is not None and str(prev_resource) != "nan":
                            event["org:resource"] = prev_resource
                        break

    overrides = _compute_active_duration_overrides(log)

    interval_log = interval_lifecycle.to_interval(log)

    df = pm4py.convert_to_dataframe(interval_log)

    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    if "start_timestamp" in df.columns:
        df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True)
    else:

        df["start_timestamp"] = df["time:timestamp"]

    if overrides:
        df = df.sort_values(["case:concept:name", "start_timestamp"],
                            kind="stable").reset_index(drop=True)

        df["_inst_idx"] = df.groupby(
            ["case:concept:name", "concept:name"]
        ).cumcount()

        keys = list(zip(df["case:concept:name"], df["concept:name"], df["_inst_idx"]))
        active_s = pd.Series(
            [overrides.get(k) for k in keys],
            index=df.index,
        )

        new_end = df["start_timestamp"] + pd.to_timedelta(active_s, unit="s")
        df["time:timestamp"] = new_end.fillna(df["time:timestamp"])
        df = df.drop(columns=["_inst_idx"])

        n_fixed = int(active_s.notna().sum())
        print(f"  Option B: corrected duration for {n_fixed} activity instance(s) "
              f"with SUSPEND/RESUME cycles")

    return df

def extract_control_flow_features(df: pd.DataFrame) -> ControlFlowFeatures:
    print("  Extracting control flow features...")
    cf = ControlFlowFeatures()
    total_cases = df["case:concept:name"].nunique()

    dfg, _, _ = pm4py.discover_dfg(df)
    cf.dfg_frequency = dict(dfg)

    cf.activity_frequency = df["concept:name"].value_counts().to_dict()

    for (a, b), count_ab in cf.dfg_frequency.items():
        count_ba = cf.dfg_frequency.get((b, a), 0)
        total = count_ab + count_ba
        cf.dependency_strength[(a, b)] = count_ab / total if total > 0 else 1.0

    variants = pm4py.get_variants_as_tuples(df)
    cf.variant_frequency = dict(variants)

    activity_case_counts = defaultdict(set)
    for case_id, group in df.groupby("case:concept:name"):
        for act in group["concept:name"].unique():
            activity_case_counts[act].add(case_id)
    cf.activity_variant_coverage = {
        act: len(cases) / total_cases
        for act, cases in activity_case_counts.items()
    }

    dfg_pairs = set(cf.dfg_frequency.keys())
    ab_before = defaultdict(int)
    ab_total  = defaultdict(int)
    for _, group in df.groupby("case:concept:name"):
        acts = group.sort_values("time:timestamp")["concept:name"].tolist()
        seen_pairs = set()
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                pair = (acts[i], acts[j])
                rev  = (acts[j], acts[i])
                if pair in dfg_pairs or rev in dfg_pairs:
                    if pair not in seen_pairs:
                        ab_total[pair] += 1
                        ab_before[pair] += 1
                        seen_pairs.add(pair)
                    if rev not in seen_pairs:
                        ab_total[rev] += 1
                        seen_pairs.add(rev)
    for pair, total in ab_total.items():
        if total > 0:
            cf.ordering_consistency[pair] = ab_before[pair] / total

    cf.rework_counts = pm4py.get_rework_cases_per_activity(df)

    return cf

def extract_time_features(df: pd.DataFrame) -> TimeFeatures:
    """All duration / waiting-time metrics rely on 'start_timestamp' (activity"""

    print("  Extracting time features...")
    tf = TimeFeatures()

    raw_duration = pm4py.get_service_time(
        df,
        start_timestamp_key="start_timestamp",
        aggregation_measure="mean",
    )
    tf.activity_duration_mean = {act: secs / 3600.0 for act, secs in raw_duration.items()}

    case_end   = df.groupby("case:concept:name")["time:timestamp"].max()
    case_start = df.groupby("case:concept:name")["start_timestamp"].min()
    case_durations_sec = (case_end - case_start).dt.total_seconds()
    tf.throughput_time_mean = (
        float(case_durations_sec.mean()) / 3600.0
        if len(case_durations_sec) > 0 else 0.0
    )

    variant_durations: dict = defaultdict(list)
    for _, group in df.groupby("case:concept:name"):
        g = group.sort_values("time:timestamp")
        dur_h = (
            g["time:timestamp"].iloc[-1] - g["start_timestamp"].iloc[0]
        ).total_seconds() / 3600.0
        variant = tuple(g["concept:name"].tolist())
        variant_durations[variant].append(dur_h)
    tf.throughput_time_per_variant = {
        v: sum(d) / len(d) for v, d in variant_durations.items()
    }

    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df_sorted["_prev_complete"] = df_sorted.groupby("case:concept:name")[
        "time:timestamp"
    ].shift(1)
    df_sorted["_prev_act"] = df_sorted.groupby("case:concept:name")[
        "concept:name"
    ].shift(1)

    mask = df_sorted["_prev_complete"].notna()
    df_w = df_sorted[mask].copy()
    df_w["_wait_h"] = (
        df_w["start_timestamp"] - df_w["_prev_complete"]
    ).dt.total_seconds().clip(lower=0) / 3600.0

    tf.waiting_time_mean = (
        df_w.groupby("concept:name")["_wait_h"].mean().to_dict()
    )

    pair_wait = df_w.groupby(["_prev_act", "concept:name"])["_wait_h"].mean()
    tf.waiting_time_between = {
        (prev, curr): val for (prev, curr), val in pair_wait.items()
    }

    for act in tf.activity_duration_mean:
        dur  = tf.activity_duration_mean.get(act, 0.0)
        wait = tf.waiting_time_mean.get(act, 0.0)
        tf.waiting_to_duration_ratio[act] = wait / dur if dur > 0 else None

    return tf

def extract_resource_features(df: pd.DataFrame) -> ResourceFeatures:
    """Resource utilization is computed vectorially (no iterrows)."""

    print("  Extracting resource features...")
    rf = ResourceFeatures()

    if "org:resource" not in df.columns:
        print("    Warning: no org:resource column, skipping resource features")
        return rf

    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df_sorted["_prev_resource"] = df_sorted.groupby("case:concept:name")[
        "org:resource"
    ].shift(1)
    df_sorted["_prev_complete"] = df_sorted.groupby("case:concept:name")[
        "time:timestamp"
    ].shift(1)
    df_sorted["_prev_act"] = df_sorted.groupby("case:concept:name")[
        "concept:name"
    ].shift(1)

    mask = df_sorted["_prev_resource"].notna()
    df_pairs = df_sorted[mask].copy()

    df_pairs["_is_handover"] = df_pairs["org:resource"] != df_pairs["_prev_resource"]

    handover_per_case = df_pairs.groupby("case:concept:name")["_is_handover"].sum()
    rf.handover_count_per_case = handover_per_case.astype(int).to_dict()
    rf.handover_count_mean = float(handover_per_case.mean()) if len(handover_per_case) > 0 else 0.0

    rf.resource_continuity_rate = float((~df_pairs["_is_handover"]).mean()) if len(df_pairs) > 0 else 0.0

    df_handover = df_pairs[df_pairs["_is_handover"]].copy()
    df_handover["_wait_h"] = (
        df_handover["start_timestamp"] - df_handover["_prev_complete"]
    ).dt.total_seconds().clip(lower=0) / 3600.0
    rf.waiting_after_handover_mean = (
        df_handover.groupby("concept:name")["_wait_h"].mean().to_dict()
    )

    repeated_rates: dict = {}
    for case_id, group in df.groupby("case:concept:name"):
        resources = group.sort_values("time:timestamp")["org:resource"].tolist()
        res_positions: dict = defaultdict(list)
        for i, r in enumerate(resources):
            res_positions[r].append(i)
        non_consec = sum(
            1
            for positions in res_positions.values()
            if len(positions) > 1
            and any(
                positions[j + 1] - positions[j] > 1
                for j in range(len(positions) - 1)
            )
        )
        total_res = len(res_positions)
        repeated_rates[str(case_id)] = non_consec / total_res if total_res > 0 else 0.0
    rf.repeated_participation_rate = repeated_rates

    obs_start = df["start_timestamp"].min()
    obs_end   = df["time:timestamp"].max()
    obs_span  = (obs_end - obs_start).total_seconds()
    if obs_span > 0:
        df["_busy_sec"] = (
            df["time:timestamp"] - df["start_timestamp"]
        ).dt.total_seconds().clip(lower=0)
        resource_busy = df.groupby("org:resource")["_busy_sec"].sum()
        rf.resource_utilization = (resource_busy / obs_span).to_dict()
        df.drop(columns=["_busy_sec"], inplace=True)

    return rf

def extract_batch_features(df: pd.DataFrame) -> BatchFeatures:
    """pm4py.discover_batches() returns:"""

    print("  Extracting batch features...")
    bf = BatchFeatures()

    if "org:resource" not in df.columns:
        print("    Warning: no org:resource column, skipping batch features")
        return bf

    try:
        batches = pm4py.discover_batches(df)
    except Exception as e:
        print(f"    Warning: batch discovery failed: {e}")
        return bf

    activity_batch_counts: dict = defaultdict(list)

    for (activity, resource), batch_count, details in batches:
        activity_batch_counts[activity].append(batch_count)

    bf.batched_activities = {
        act: sum(counts) for act, counts in activity_batch_counts.items()
    }

    for activity in bf.batched_activities:
        act_df       = df[df["concept:name"] == activity]
        total_events = len(act_df)
        total_batches = bf.batched_activities[activity]
        if total_batches > 0:
            bf.batch_size_mean[activity] = total_events / total_batches

    for activity in bf.batched_activities:
        act_df     = df[df["concept:name"] == activity].sort_values("start_timestamp")
        timestamps = act_df["start_timestamp"].tolist()
        if len(timestamps) < 3:
            bf.periodicity_strength[activity] = 0.0
            continue
        gaps = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap == 0:
            bf.periodicity_strength[activity] = 0.0
            continue
        std_gap = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
        cv = std_gap / mean_gap
        bf.periodicity_strength[activity] = max(0.0, 1.0 - cv)
        bf.batch_interval_mean[activity]  = mean_gap / 3600.0

    return bf

def extract_process_graph_features(bpmn_path: str) -> ProcessGraphFeatures:
    """Reads a BPMN file and builds a NetworkX DiGraph."""

    print("  Extracting process graph features...")
    pgf = ProcessGraphFeatures()

    try:
        bpmn_model = pm4py.read_bpmn(bpmn_path)
    except Exception as e:
        print(f"    Warning: could not read BPMN model: {e}")
        return pgf

    G = nx.DiGraph()

    for node in bpmn_model.get_nodes():
        node_id   = str(id(node))
        node_name = node.get_name() or ""

        if isinstance(node, (BPMN.Task, BPMN.UserTask, BPMN.SendTask, BPMN.SubProcess)):
            node_type = "activity"
        elif isinstance(node, (BPMN.ExclusiveGateway,)):
            node_type = "xor_gateway"
        elif isinstance(node, (BPMN.ParallelGateway,)):
            node_type = "and_gateway"
        elif isinstance(node, (BPMN.InclusiveGateway,)):
            node_type = "or_gateway"
        elif isinstance(node, (BPMN.EventBasedGateway,)):
            node_type = "event_gateway"
        elif isinstance(node, BPMN.Event):
            node_type = "event"
        else:
            node_type = "unknown"

        G.add_node(node_id, name=node_name, type=node_type)
        pgf.node_metadata[node_id] = {"name": node_name, "type": node_type}
        if node_type == "activity" and node_name:
            pgf.activity_to_node[node_name] = node_id

    for flow in bpmn_model.get_flows():

        if not isinstance(flow, BPMN.SequenceFlow):
            continue
        src = str(id(flow.get_source()))
        tgt = str(id(flow.get_target()))
        if src in G.nodes and tgt in G.nodes:
            G.add_edge(src, tgt)

    pgf.graph = G
    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return pgf

def extract_features(log_path: str, bpmn_path: str) -> FeatureStore:
    print(f"\nLoading event log: {log_path}")

    df = prepare_dataframe(log_path)

    print(f"  Cases     : {df['case:concept:name'].nunique()}")
    print(f"  Events    : {len(df)}")
    print(f"  Activities: {df['concept:name'].nunique()}")
    print(f"  Has start_timestamp: {'start_timestamp' in df.columns}")

    store = FeatureStore()
    store.total_cases = df["case:concept:name"].nunique()
    store.activities  = df["concept:name"].unique().tolist()

    print("\nExtracting features...")
    store.control_flow  = extract_control_flow_features(df)
    store.time          = extract_time_features(df)
    store.resource      = extract_resource_features(df)
    store.batch         = extract_batch_features(df)
    store.process_graph = extract_process_graph_features(bpmn_path)

    return store

def print_summary(store: FeatureStore) -> None:
    print("\n" + "=" * 60)
    print("FEATURE STORE SUMMARY")
    print("=" * 60)
    print(f"Total cases : {store.total_cases}")
    print(f"Activities  : {len(store.activities)}")

    print("\n── Control Flow ──")
    print(f"  DFG edges       : {len(store.control_flow.dfg_frequency)}")
    print(f"  Unique variants : {len(store.control_flow.variant_frequency)}")
    if store.control_flow.variant_frequency:
        top   = max(store.control_flow.variant_frequency, key=store.control_flow.variant_frequency.get)
        total = sum(store.control_flow.variant_frequency.values())
        print(f"  Dominant variant coverage: "
              f"{store.control_flow.variant_frequency[top] / total:.1%}")
    if store.control_flow.rework_counts:
        top_rework = max(store.control_flow.rework_counts, key=store.control_flow.rework_counts.get)
        print(f"  Most reworked activity: {top_rework!r} "
              f"({store.control_flow.rework_counts[top_rework]} cases)")

    print("\n── Time ──")
    print(f"  Mean throughput time : {store.time.throughput_time_mean:.1f} h")
    if store.time.waiting_time_mean:
        top = max(store.time.waiting_time_mean, key=store.time.waiting_time_mean.get)
        print(f"  Highest waiting time : {top!r} "
              f"({store.time.waiting_time_mean[top]:.1f} h)")
    if store.time.activity_duration_mean:
        top = max(store.time.activity_duration_mean, key=store.time.activity_duration_mean.get)
        print(f"  Longest activity     : {top!r} "
              f"({store.time.activity_duration_mean[top]:.2f} h)")
    valid_ratios = {k: v for k, v in store.time.waiting_to_duration_ratio.items() if v is not None}
    if valid_ratios:
        top = max(valid_ratios, key=valid_ratios.get)
        print(f"  Highest wait/dur ratio: {top!r} ({valid_ratios[top]:.1f}x)")

    print("\n── Resource ──")
    print(f"  Mean handovers/case  : {store.resource.handover_count_mean:.1f}")
    print(f"  Resource continuity  : {store.resource.resource_continuity_rate:.1%}")
    if store.resource.resource_utilization:
        top = max(store.resource.resource_utilization, key=store.resource.resource_utilization.get)
        print(f"  Most utilized resource: {top!r} "
              f"({store.resource.resource_utilization[top]:.1%})")

    print("\n── Batch ──")
    print(f"  Batched activities   : {len(store.batch.batched_activities)}")
    if store.batch.batched_activities:
        top = max(store.batch.batched_activities, key=store.batch.batched_activities.get)
        print(f"  Most batched         : {top!r} "
              f"({store.batch.batched_activities[top]} batches)")

    print("\n── Process Graph ──")
    if store.process_graph.graph:
        G = store.process_graph.graph
        print(f"  Nodes : {G.number_of_nodes()}")
        print(f"  Edges : {G.number_of_edges()}")
        type_counts: dict = defaultdict(int)
        for _, data in G.nodes(data=True):
            type_counts[data.get("type", "unknown")] += 1
        for ntype, count in sorted(type_counts.items()):
            print(f"    {ntype}: {count}")
