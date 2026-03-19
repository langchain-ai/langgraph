#[cfg(feature = "python-bindings")]
use crate::engine::{
    run_graph_with_callback, run_loop_block_on, AnyOfCondition, Engine, NodeExecResult,
    NodeOutcome, SendPayload, WaitCondition, WaitEvent, WaitRequest,
};
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyAny;
#[cfg(feature = "python-bindings")]
use pyo3::types::{PyDict, PyList, PyTuple};
#[cfg(feature = "python-bindings")]
use serde_json::Value;
#[cfg(feature = "python-bindings")]
use std::sync::Arc;

#[cfg(feature = "python-bindings")]
#[pyclass]
struct PyRustEngine {
    inner: Engine,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyRustEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: Engine::new(),
        }
    }

    fn add_async_channel(&self, name: &str) {
        self.inner.add_async_channel(name);
    }

    fn add_custom_output_stream(&self, stream_name: &str) -> PyResult<()> {
        self.inner
            .add_custom_output_stream(stream_name)
            .map_err(PyValueError::new_err)
    }

    fn publish_obj(&self, py: Python<'_>, channel: &str, value: Py<PyAny>) -> PyResult<()> {
        let value_json = py_obj_to_json_string(py, &value.bind(py))?;
        let parsed: Value = serde_json::from_str(&value_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid Python JSON value: {e}")))?;
        self.inner
            .publish_json(channel, parsed)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (channel, min, max=None))]
    fn wait_channel(
        &self,
        py: Python<'_>,
        channel: &str,
        min: usize,
        max: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let cond = WaitCondition::Channel {
            channel: channel.to_string(),
            min,
            max: max.unwrap_or(0),
        };
        let any_of = AnyOfCondition {
            conditions: vec![cond],
        };
        let event = run_loop_block_on(self.inner.wait_for_any_of_async(&any_of))
            .map_err(PyValueError::new_err)?;
        let event_json = serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
        json_string_to_py_obj(py, &event_json)
    }

    fn wait_timer(&self, py: Python<'_>, seconds: f64) -> PyResult<Py<PyAny>> {
        let cond = WaitCondition::Timer { seconds };
        let any_of = AnyOfCondition {
            conditions: vec![cond],
        };
        let event = run_loop_block_on(self.inner.wait_for_any_of_async(&any_of))
            .map_err(PyValueError::new_err)?;
        let event_json = serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
        json_string_to_py_obj(py, &event_json)
    }

    fn wait_any_of_obj(&self, py: Python<'_>, any_of_payload: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let payload_json = py_obj_to_json_string(py, &any_of_payload.bind(py))?;
        let event_json = self._wait_any_of_json(&payload_json)?;
        json_string_to_py_obj(py, &event_json)
    }

    fn _wait_any_of_json(&self, any_of_json: &str) -> PyResult<String> {
        let any_of: AnyOfCondition = serde_json::from_str(any_of_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid any_of JSON: {e}")))?;
        let event = run_loop_block_on(self.inner.wait_for_any_of_async(&any_of))
            .map_err(PyValueError::new_err)?;
        serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))
    }

    fn wait_condition_json(&self, cond_json: &str) -> PyResult<String> {
        let cond: WaitCondition = serde_json::from_str(cond_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid condition JSON: {e}")))?;
        let any_of = AnyOfCondition {
            conditions: vec![cond],
        };
        let event = run_loop_block_on(self.inner.wait_for_any_of_async(&any_of))
            .map_err(PyValueError::new_err)?;
        serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))
    }

    #[pyo3(signature = (stream_mode=None))]
    fn start_stream(&self, stream_mode: Option<&str>) -> PyResult<()> {
        self.inner
            .start_stream(stream_mode)
            .map_err(PyValueError::new_err)
    }

    fn receive_stream_obj(&self, py: Python<'_>, stream_name: &str) -> PyResult<Py<PyAny>> {
        let event =
            py.allow_threads(|| run_loop_block_on(self.inner.receive_stream_async(stream_name)));
        match event {
            Some(value) => {
                let event_json = serde_json::to_string(&value)
                    .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
                json_string_to_py_obj(py, &event_json)
            }
            None => Ok(py.None()),
        }
    }

    fn send_custom_stream_event_obj(
        &self,
        py: Python<'_>,
        stream_name: &str,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        let value_json = py_obj_to_json_string(py, &value.bind(py))?;
        let parsed: Value = serde_json::from_str(&value_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid Python JSON value: {e}")))?;
        self.inner.send_custom_stream_event(stream_name, parsed);
        Ok(())
    }

    fn close_all_streams(&self) {
        self.inner.close_all_streams();
    }

    #[pyo3(signature = (entry_point, finish_point, initial_state, callback, stream_mode=None))]
    fn run_graph_py(
        &self,
        py: Python<'_>,
        entry_point: &str,
        finish_point: &str,
        initial_state: Py<PyAny>,
        callback: Py<PyAny>,
        stream_mode: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(mode) = stream_mode {
            self.inner
                .start_stream(Some(mode))
                .map_err(PyValueError::new_err)?;
        }

        let engine = self.inner.clone();
        let callback = Arc::new(callback);
        let entry_point = entry_point.to_string();
        let finish_point = finish_point.to_string();
        let initial_state = Arc::new(initial_state);
        let initial_input = Arc::clone(&initial_state);

        let run_result =
            py.allow_threads(move || {
                run_loop_block_on(run_graph_with_callback(
                    entry_point,
                    finish_point,
                    initial_state,
                    initial_input,
                    engine.clone(),
                    {
                        let callback = Arc::clone(&callback);
                    move |node: String,
                          arg: Arc<Py<PyAny>>,
                          state_snapshot: Arc<Py<PyAny>>|
                          -> Result<NodeOutcome<Arc<Py<PyAny>>, Arc<Py<PyAny>>>, String> {
                        Python::with_gil(|py| -> Result<NodeOutcome<Arc<Py<PyAny>>, Arc<Py<PyAny>>>, String> {
                            let callback_bound = callback.as_ref().bind(py);
                            let payload_obj = callback_bound
                                .call1((
                                    node.as_str(),
                                    arg.as_ref().clone_ref(py),
                                    state_snapshot.as_ref().clone_ref(py),
                                ))
                                .map_err(|e| format!("callback failed for node `{node}`: {e}"))?;
                            parse_node_outcome_arc(py, &payload_obj).map_err(|e| {
                                format!("invalid callback payload for `{node}`: {e}")
                            })
                        })
                    }
                    },
                |state: &mut Arc<Py<PyAny>>, update: Option<Arc<Py<PyAny>>>| -> Result<(), String> {
                        Python::with_gil(|py| -> Result<(), String> {
                            if let Some(update) = update {
                            apply_update_to_state(py, state.as_ref(), update.as_ref())
                                    .map_err(|e| format!("state merge failed: {e}"))?;
                            }
                            Ok(())
                        })
                    },
                |arg: Arc<Py<PyAny>>, event: WaitEvent| -> Result<Arc<Py<PyAny>>, String> {
                    wrap_resume_arg(arg.as_ref(), &event).map(Arc::new)
                    },
                ))
            });

        self.inner.close_all_streams();
        let out = run_result.map_err(PyValueError::new_err)?;
        Ok(out.as_ref().clone_ref(py))
    }
}

#[cfg(feature = "python-bindings")]
fn parse_node_outcome_arc(
    py: Python<'_>,
    payload_obj: &Bound<'_, PyAny>,
) -> Result<NodeOutcome<Arc<Py<PyAny>>, Arc<Py<PyAny>>>, String> {
    let payload_dict = payload_obj
        .downcast::<PyDict>()
        .map_err(|_| "payload must be a dict".to_string())?;
    let suspended_item = payload_dict
        .get_item("suspend")
        .map_err(|e| format!("failed to read suspend: {e}"))?;
    if let Some(wait_obj) = suspended_item {
        let wait_json = py_obj_to_json_string(py, &wait_obj)
            .map_err(|e| format!("failed to encode suspend payload: {e}"))?;
        let wait: WaitRequest = serde_json::from_str(&wait_json)
            .map_err(|e| format!("invalid suspend payload: {e}"))?;
        return Ok(NodeOutcome::Suspended { wait });
    }

    let update_item = payload_dict
        .get_item("update")
        .map_err(|e| format!("failed to read update: {e}"))?;
    let update = match update_item {
        Some(v) if !v.is_none() => Some(Arc::new(v.unbind())),
        _ => None,
    };

    let sends_obj = payload_dict
        .get_item("sends")
        .map_err(|e| format!("failed to read sends: {e}"))?
        .ok_or_else(|| "missing sends".to_string())?;
    let sends_list = sends_obj
        .downcast::<PyList>()
        .map_err(|_| "sends must be a list".to_string())?;

    let mut sends = Vec::with_capacity(sends_list.len());
    for item in sends_list.iter() {
        let send_dict = item
            .downcast::<PyDict>()
            .map_err(|_| "send item must be a dict".to_string())?;
        let node_obj = send_dict
            .get_item("node")
            .map_err(|e| format!("failed to read send.node: {e}"))?
            .ok_or_else(|| "send.node is required".to_string())?;
        let node = node_obj
            .extract::<String>()
            .map_err(|e| format!("send.node must be string: {e}"))?;
        let arg: Arc<Py<PyAny>> = match send_dict.get_item("arg") {
            Ok(Some(v)) => Arc::new(v.unbind()),
            Ok(None) => Arc::new(Python::with_gil(|py| py.None())),
            Err(e) => return Err(format!("failed to read send.arg: {e}")),
        };
        sends.push(SendPayload { node, arg });
    }

    Ok(NodeOutcome::Completed(NodeExecResult { update, sends }))
}

#[cfg(feature = "python-bindings")]
fn wrap_resume_arg(arg: &Py<PyAny>, event: &WaitEvent) -> Result<Py<PyAny>, String> {
    Python::with_gil(|py| -> Result<Py<PyAny>, String> {
        let wrapper = PyDict::new(py);
        wrapper
            .set_item("__lg_resume_arg__", arg.clone_ref(py))
            .map_err(|e| format!("failed to set resume arg: {e}"))?;
        let event_json = serde_json::to_string(event)
            .map_err(|e| format!("failed to encode wait event: {e}"))?;
        let event_obj = json_string_to_py_obj(py, &event_json)
            .map_err(|e| format!("failed to parse event: {e}"))?;
        wrapper
            .set_item("__lg_resume_event__", event_obj.bind(py))
            .map_err(|e| format!("failed to set resume event: {e}"))?;
        Ok(wrapper.unbind().into_any())
    })
}

#[cfg(feature = "python-bindings")]
fn apply_update_to_state(py: Python<'_>, state: &Py<PyAny>, update: &Py<PyAny>) -> PyResult<()> {
    let state_obj = state.bind(py);
    let update_obj = update.bind(py);

    if update_obj.is_none() {
        return Ok(());
    }
    if state_obj.is_instance_of::<PyDict>() && update_obj.is_instance_of::<PyDict>() {
        let state_dict = state_obj.downcast::<PyDict>()?;
        let update_dict = update_obj.downcast::<PyDict>()?;
        state_dict.call_method1("update", (update_dict,))?;
        return Ok(());
    }
    if let Ok(tuple_like) = update_obj.downcast::<PyList>() {
        apply_pair_updates(state_obj, tuple_like)?;
        return Ok(());
    }
    if let Ok(tuple_like) = update_obj.downcast::<PyTuple>() {
        let list = PyList::new(py, tuple_like)?;
        apply_pair_updates(state_obj, &list)?;
    }
    Ok(())
}

#[cfg(feature = "python-bindings")]
fn apply_pair_updates(state_obj: &Bound<'_, PyAny>, entries: &Bound<'_, PyList>) -> PyResult<()> {
    if !state_obj.is_instance_of::<PyDict>() {
        return Ok(());
    }
    let state_dict = state_obj.downcast::<PyDict>()?;
    for entry in entries.iter() {
        if let Ok(pair) = entry.downcast::<PyTuple>() {
            if pair.len() == 2 {
                let key_obj = pair.get_item(0)?;
                if let Ok(key) = key_obj.extract::<String>() {
                    let value_obj = pair.get_item(1)?;
                    state_dict.set_item(key, value_obj)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "python-bindings")]
fn py_obj_to_json_string(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let json_mod = py.import("json")?;
    let dumped = json_mod.call_method1("dumps", (obj,))?;
    dumped.extract::<String>()
}

#[cfg(feature = "python-bindings")]
fn json_string_to_py_obj(py: Python<'_>, value: &str) -> PyResult<Py<PyAny>> {
    let json_mod = py.import("json")?;
    let loaded = json_mod.call_method1("loads", (value,))?;
    Ok(loaded.unbind())
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn langgraph_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRustEngine>()?;
    Ok(())
}
