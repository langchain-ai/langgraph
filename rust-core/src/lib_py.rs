#[cfg(feature = "python-bindings")]
use crate::engine::{
    node_pool_execute, run_loop_pool_execute, run_scheduler_loop, AnyOfCondition, Engine,
    NodeExecResult, SendPayload, WaitCondition,
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
use std::sync::mpsc;
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

    fn publish_json(&self, channel: &str, value_json: &str) -> PyResult<()> {
        let value: Value = serde_json::from_str(value_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON value: {e}")))?;
        self.inner
            .publish_json(channel, value)
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

    fn wait_any_of_json(&self, any_of_json: &str) -> PyResult<String> {
        let any_of: AnyOfCondition = serde_json::from_str(any_of_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid any_of JSON: {e}")))?;
        let event = self
            .inner
            .wait_for_any_of(&any_of)
            .map_err(PyValueError::new_err)?;
        serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))
    }

    fn wait_channel(&self, py: Python<'_>, channel: &str, n: usize) -> PyResult<Py<PyAny>> {
        let cond = WaitCondition::Channel {
            channel: channel.to_string(),
            n,
        };
        let event = self.inner.wait_for(&cond).map_err(PyValueError::new_err)?;
        let event_json = serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
        json_string_to_py_obj(py, &event_json)
    }

    fn wait_timer(&self, py: Python<'_>, seconds: f64) -> PyResult<Py<PyAny>> {
        let cond = WaitCondition::Timer { seconds };
        let event = self.inner.wait_for(&cond).map_err(PyValueError::new_err)?;
        let event_json = serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
        json_string_to_py_obj(py, &event_json)
    }

    fn wait_any_of_obj(&self, py: Python<'_>, any_of_payload: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let payload_json = py_obj_to_json_string(py, &any_of_payload.bind(py))?;
        let any_of: AnyOfCondition = serde_json::from_str(&payload_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid any_of payload: {e}")))?;
        let event = self
            .inner
            .wait_for_any_of(&any_of)
            .map_err(PyValueError::new_err)?;
        let event_json = serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
        json_string_to_py_obj(py, &event_json)
    }

    fn wait_condition_json(&self, cond_json: &str) -> PyResult<String> {
        let cond: WaitCondition = serde_json::from_str(cond_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid condition JSON: {e}")))?;
        let event = self.inner.wait_for(&cond).map_err(PyValueError::new_err)?;
        serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))
    }

    fn run_graph_py(
        &self,
        py: Python<'_>,
        entry_point: &str,
        finish_point: &str,
        initial_state: Py<PyAny>,
        callback: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let state = Arc::new(initial_state);
        let callback = Arc::new(callback);
        let entry_point = entry_point.to_string();
        let finish_point = finish_point.to_string();
        let (done_tx, done_rx) = mpsc::channel::<Result<(), String>>();
        let state_for_run = Arc::clone(&state);
        run_loop_pool_execute(move || {
            let run_result =
                run_graph_scheduler(entry_point, finish_point, callback, state_for_run);
            let _ = done_tx.send(run_result);
        })
        .map_err(PyValueError::new_err)?;
        let run_result = py
            .allow_threads(move || done_rx.recv())
            .map_err(|e| PyValueError::new_err(format!("run-loop recv failed: {e}")))?;
        run_result.map_err(PyValueError::new_err)?;
        Ok((*state).clone_ref(py))
    }
}

#[cfg(feature = "python-bindings")]
fn run_graph_scheduler(
    entry_point: String,
    finish_point: String,
    callback: Arc<Py<PyAny>>,
    state: Arc<Py<PyAny>>,
) -> Result<(), String> {
    let (tx, rx) =
        mpsc::channel::<Result<(String, NodeExecResult<Py<PyAny>, Py<PyAny>>), String>>();
    let initial_arg = Python::with_gil(|py| (*state).clone_ref(py));
    let callback_for_spawn = Arc::clone(&callback);
    let state_for_spawn = Arc::clone(&state);
    let tx_for_spawn = tx.clone();
    let state_for_merge = Arc::clone(&state);
    run_scheduler_loop(
        entry_point,
        &finish_point,
        initial_arg,
        move |node, arg| {
            spawn_node_task(
                node,
                arg,
                tx_for_spawn.clone(),
                Arc::clone(&callback_for_spawn),
                Arc::clone(&state_for_spawn),
            )
        },
        move |node_name, update| {
            Python::with_gil(|py| -> Result<(), String> {
                if let Some(update) = update {
                    apply_update_to_state(py, state_for_merge.as_ref(), &update)
                        .map_err(|e| format!("state merge failed for `{node_name}`: {e}"))?;
                }
                Ok(())
            })
        },
        rx,
    )
}

#[cfg(feature = "python-bindings")]
fn spawn_node_task(
    node: String,
    arg: Py<PyAny>,
    tx: mpsc::Sender<Result<(String, NodeExecResult<Py<PyAny>, Py<PyAny>>), String>>,
    callback: Arc<Py<PyAny>>,
    state_for_task: Arc<Py<PyAny>>,
) -> Result<(), String> {
    node_pool_execute(move || {
        let outcome = Python::with_gil(
            |py| -> Result<(String, NodeExecResult<Py<PyAny>, Py<PyAny>>), String> {
                let callback_bound = callback.as_ref().bind(py);
                let payload_obj = callback_bound
                    .call1((node.as_str(), arg, (*state_for_task).clone_ref(py)))
                    .map_err(|e| format!("callback failed for node `{node}`: {e}"))?;
                let payload = parse_node_exec_result(&payload_obj)
                    .map_err(|e| format!("invalid callback payload for `{node}`: {e}"))?;
                Ok((node, payload))
            },
        );
        let _ = tx.send(outcome);
    })
}

#[cfg(feature = "python-bindings")]
fn parse_node_exec_result(
    payload_obj: &Bound<'_, PyAny>,
) -> Result<NodeExecResult<Py<PyAny>, Py<PyAny>>, String> {
    let payload_dict = payload_obj
        .downcast::<PyDict>()
        .map_err(|_| "payload must be a dict".to_string())?;

    let update_item = payload_dict
        .get_item("update")
        .map_err(|e| format!("failed to read update: {e}"))?;
    let update = match update_item {
        Some(v) if !v.is_none() => Some(v.unbind()),
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
        let arg: Py<PyAny> = match send_dict.get_item("arg") {
            Ok(Some(v)) => v.unbind(),
            Ok(None) => Python::with_gil(|py| py.None()),
            Err(e) => return Err(format!("failed to read send.arg: {e}")),
        };
        sends.push(SendPayload { node, arg });
    }

    Ok(NodeExecResult { update, sends })
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
