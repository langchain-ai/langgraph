#[cfg(feature = "python-bindings")]
use crate::engine::{
    node_pool_execute, run_loop_block_on, run_loop_spawn, AnyOfCondition, Engine, NodeExecResult,
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
use std::sync::mpsc;
#[cfg(feature = "python-bindings")]
use std::sync::Arc;
#[cfg(feature = "python-bindings")]
use tokio::sync::mpsc as tokio_mpsc;

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
        let event = run_loop_block_on(self.inner.wait_for_any_of_async(&any_of))
            .map_err(PyValueError::new_err)?;
        serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))
    }

    fn wait_channel(&self, py: Python<'_>, channel: &str, n: usize) -> PyResult<Py<PyAny>> {
        let cond = WaitCondition::Channel {
            channel: channel.to_string(),
            n,
        };
        let event =
            run_loop_block_on(self.inner.wait_for_async(&cond)).map_err(PyValueError::new_err)?;
        let event_json = serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
        json_string_to_py_obj(py, &event_json)
    }

    fn wait_timer(&self, py: Python<'_>, seconds: f64) -> PyResult<Py<PyAny>> {
        let cond = WaitCondition::Timer { seconds };
        let event =
            run_loop_block_on(self.inner.wait_for_async(&cond)).map_err(PyValueError::new_err)?;
        let event_json = serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
        json_string_to_py_obj(py, &event_json)
    }

    fn wait_any_of_obj(&self, py: Python<'_>, any_of_payload: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let payload_json = py_obj_to_json_string(py, &any_of_payload.bind(py))?;
        let any_of: AnyOfCondition = serde_json::from_str(&payload_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid any_of payload: {e}")))?;
        let event = run_loop_block_on(self.inner.wait_for_any_of_async(&any_of))
            .map_err(PyValueError::new_err)?;
        let event_json = serde_json::to_string(&event)
            .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
        json_string_to_py_obj(py, &event_json)
    }

    fn wait_condition_json(&self, cond_json: &str) -> PyResult<String> {
        let cond: WaitCondition = serde_json::from_str(cond_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid condition JSON: {e}")))?;
        let event =
            run_loop_block_on(self.inner.wait_for_async(&cond)).map_err(PyValueError::new_err)?;
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
        let engine = self.inner.clone();
        run_loop_spawn(async move {
            let run_result =
                run_graph_scheduler(entry_point, finish_point, callback, state_for_run, engine)
                    .await;
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
enum SchedulerEventPy {
    Node(Result<NodeExecutionPy, String>),
    Resume {
        node: String,
        arg: Py<PyAny>,
        event: WaitEvent,
    },
    WaitError(String),
}

struct NodeExecutionPy {
    node: String,
    arg: Py<PyAny>,
    outcome: NodeOutcome<Py<PyAny>, Py<PyAny>>,
}

#[cfg(feature = "python-bindings")]
async fn run_graph_scheduler(
    entry_point: String,
    finish_point: String,
    callback: Arc<Py<PyAny>>,
    state: Arc<Py<PyAny>>,
    engine: Engine,
) -> Result<(), String> {
    let (tx, mut rx) = tokio_mpsc::unbounded_channel::<SchedulerEventPy>();
    let initial_arg = Python::with_gil(|py| (*state).clone_ref(py));
    let callback_for_spawn = Arc::clone(&callback);
    let state_for_spawn = Arc::clone(&state);
    let tx_for_spawn = tx.clone();
    let state_for_merge = Arc::clone(&state);
    let mut active: usize = 1;
    let mut waiting: usize = 0;
    spawn_node_task(
        entry_point,
        initial_arg,
        tx_for_spawn.clone(),
        Arc::clone(&callback_for_spawn),
        Arc::clone(&state_for_spawn),
    )?;

    while active > 0 || waiting > 0 {
        let event = rx
            .recv()
            .await
            .ok_or_else(|| "scheduler event channel closed".to_string())?;
        match event {
            SchedulerEventPy::Node(result) => {
                active = active.saturating_sub(1);
                let exec = result?;
                match exec.outcome {
                    NodeOutcome::Completed(node_result) => {
                        Python::with_gil(|py| -> Result<(), String> {
                            if let Some(update) = node_result.update {
                                apply_update_to_state(py, state_for_merge.as_ref(), &update)
                                    .map_err(|e| {
                                        format!("state merge failed for `{}`: {e}", exec.node)
                                    })?;
                            }
                            Ok(())
                        })?;
                        if exec.node == finish_point {
                            break;
                        }
                        for send in node_result.sends {
                            active += 1;
                            spawn_node_task(
                                send.node,
                                send.arg,
                                tx_for_spawn.clone(),
                                Arc::clone(&callback_for_spawn),
                                Arc::clone(&state_for_spawn),
                            )?;
                        }
                    }
                    NodeOutcome::Suspended { wait } => {
                        waiting += 1;
                        let tx_wait = tx_for_spawn.clone();
                        let node = exec.node;
                        let arg = exec.arg;
                        let engine_for_wait = engine.clone();
                        tokio::spawn(async move {
                            let outcome = engine_for_wait.wait_request_async(&wait).await;
                            match outcome {
                                Ok(event) => {
                                    let _ = tx_wait.send(SchedulerEventPy::Resume { node, arg, event });
                                }
                                Err(e) => {
                                    let _ = tx_wait.send(SchedulerEventPy::WaitError(e));
                                }
                            }
                        });
                    }
                }
            }
            SchedulerEventPy::Resume { node, arg, event } => {
                waiting = waiting.saturating_sub(1);
                active += 1;
                let resume_arg = wrap_resume_arg(&arg, &event)?;
                spawn_node_task(
                    node,
                    resume_arg,
                    tx_for_spawn.clone(),
                    Arc::clone(&callback_for_spawn),
                    Arc::clone(&state_for_spawn),
                )?;
            }
            SchedulerEventPy::WaitError(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(feature = "python-bindings")]
fn spawn_node_task(
    node: String,
    arg: Py<PyAny>,
    tx: tokio_mpsc::UnboundedSender<SchedulerEventPy>,
    callback: Arc<Py<PyAny>>,
    state_for_task: Arc<Py<PyAny>>,
) -> Result<(), String> {
    node_pool_execute(move || {
        let node_for_result = node.clone();
        let arg_for_result = Python::with_gil(|py| arg.clone_ref(py));
        let outcome = Python::with_gil(
            |py| -> Result<NodeExecutionPy, String> {
                let callback_bound = callback.as_ref().bind(py);
                let payload_obj = callback_bound
                    .call1((node.as_str(), arg, (*state_for_task).clone_ref(py)))
                    .map_err(|e| format!("callback failed for node `{node}`: {e}"))?;
                let payload = parse_node_outcome(py, &payload_obj)
                    .map_err(|e| format!("invalid callback payload for `{node}`: {e}"))?;
                Ok(NodeExecutionPy {
                    node: node_for_result,
                    arg: arg_for_result,
                    outcome: payload,
                })
            },
        );
        let _ = tx.send(SchedulerEventPy::Node(outcome));
    })
}

#[cfg(feature = "python-bindings")]
fn parse_node_outcome(
    py: Python<'_>,
    payload_obj: &Bound<'_, PyAny>,
) -> Result<NodeOutcome<Py<PyAny>, Py<PyAny>>, String> {
    let payload_dict = payload_obj
        .downcast::<PyDict>()
        .map_err(|_| "payload must be a dict".to_string())?;
    let suspended_item = payload_dict
        .get_item("suspend")
        .map_err(|e| format!("failed to read suspend: {e}"))?;
    if let Some(wait_obj) = suspended_item {
        let wait_json = py_obj_to_json_string(py, &wait_obj)
            .map_err(|e| format!("failed to encode suspend payload: {e}"))?;
        let wait: WaitRequest =
            serde_json::from_str(&wait_json).map_err(|e| format!("invalid suspend payload: {e}"))?;
        return Ok(NodeOutcome::Suspended { wait });
    }

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

    Ok(NodeOutcome::Completed(NodeExecResult { update, sends }))
}

#[cfg(feature = "python-bindings")]
fn wrap_resume_arg(arg: &Py<PyAny>, event: &WaitEvent) -> Result<Py<PyAny>, String> {
    Python::with_gil(|py| -> Result<Py<PyAny>, String> {
        let wrapper = PyDict::new(py);
        wrapper
            .set_item("__lg_resume_arg__", arg.clone_ref(py))
            .map_err(|e| format!("failed to set resume arg: {e}"))?;
        let event_json =
            serde_json::to_string(event).map_err(|e| format!("failed to encode wait event: {e}"))?;
        let event_obj =
            json_string_to_py_obj(py, &event_json).map_err(|e| format!("failed to parse event: {e}"))?;
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
