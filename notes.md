* move `_warn_invalid_state_schema` to private utils
* move `_get_node_name` to private utils
* make `StateNodeSpec` private, make it generic on node input type
* improve all docstrings in `state.py` (and other places, but this one is important)
* helper deprecation method in `state.py`
* More useful example in state graph API docs
* schema to mapper stuff seems funky
* `CompiledStateGraph` should have a full signature
* Create a `GraphSettings` structure managed in pregel
* Huge change, but would like to move away from `configurable` being where we store all of this internal stuff.
* Should `deprecated_kwargs` be of type never?