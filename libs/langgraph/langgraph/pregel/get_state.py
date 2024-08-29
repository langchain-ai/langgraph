from langgraph.constants import NS_SEP
from langgraph.pregel.types import StateSnapshot


def assemble_state_snapshot_hierarchy(
    root_checkpoint_ns: str,
    checkpoint_ns_to_state_snapshots: dict[str, StateSnapshot],
) -> StateSnapshot:
    checkpoint_ns_list_to_visit = sorted(
        checkpoint_ns_to_state_snapshots.keys(),
        key=lambda x: len(x.split(NS_SEP)),
    )
    while checkpoint_ns_list_to_visit:
        checkpoint_ns = checkpoint_ns_list_to_visit.pop()
        state_snapshot = checkpoint_ns_to_state_snapshots[checkpoint_ns]
        *path, subgraph_node = checkpoint_ns.split(NS_SEP)
        parent_checkpoint_ns = NS_SEP.join(path)
        if subgraph_node and (
            parent_state_snapshot := checkpoint_ns_to_state_snapshots.get(
                parent_checkpoint_ns
            )
        ):
            parent_subgraph_snapshots = {
                **(parent_state_snapshot.subgraphs or {}),
                subgraph_node: state_snapshot,
            }
            checkpoint_ns_to_state_snapshots[parent_checkpoint_ns] = (
                checkpoint_ns_to_state_snapshots[
                    parent_checkpoint_ns
                ]._replace(subgraphs=parent_subgraph_snapshots)
            )

    state_snapshot = checkpoint_ns_to_state_snapshots.pop(root_checkpoint_ns, None)
    if state_snapshot is None:
        raise ValueError(f"Missing checkpoint for checkpoint NS '{root_checkpoint_ns}'")
    return state_snapshot
