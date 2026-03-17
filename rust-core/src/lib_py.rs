#[cfg(feature = "python-bindings")]
use crate::engine::{
    parse_callback_envelope_json, run_graph_json_with_callback, run_loop_block_on, AnyOfCondition,
    Engine, NodeOutcome, WaitCondition,
};
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyAny;
#[cfg(feature = "python-bindings")]
use serde_json::Value;

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
        self.publish_json(channel, &value_json)
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
        let event_json = self.wait_any_of_json(&payload_json)?;
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

    #[pyo3(signature = (stream_mode=None))]
    fn start_stream(&self, stream_mode: Option<&str>) -> PyResult<()> {
        self.inner
            .start_stream(stream_mode)
            .map_err(PyValueError::new_err)
    }

    fn receive_stream_obj(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match run_loop_block_on(self.inner.receive_stream_async()) {
            Some(value) => {
                let event_json = serde_json::to_string(&value)
                    .map_err(|e| PyValueError::new_err(format!("Serialize event failed: {e}")))?;
                json_string_to_py_obj(py, &event_json)
            }
            None => Ok(py.None()),
        }
    }

    fn send_custom_stream_event_obj(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let value_json = py_obj_to_json_string(py, &value.bind(py))?;
        let parsed: Value = serde_json::from_str(&value_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid Python JSON value: {e}")))?;
        self.inner.send_custom_stream_event(parsed);
        Ok(())
    }

    fn close_stream(&self) {
        self.inner.close_stream();
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
        let state_json = py_obj_to_json_string(py, &initial_state.bind(py))?;
        let initial_state_value: Value = serde_json::from_str(&state_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid initial state: {e}")))?;

        self.inner
            .start_stream(stream_mode)
            .map_err(PyValueError::new_err)?;

        let callback_arc = std::sync::Arc::new(callback);
        let run_result = run_loop_block_on(run_graph_json_with_callback(
            entry_point.to_string(),
            finish_point.to_string(),
            initial_state_value.clone(),
            initial_state_value,
            self.inner.clone(),
            move |node: String,
                  arg: Value,
                  state_snapshot: Value|
                  -> Result<NodeOutcome<Value, Value>, String> {
                Python::with_gil(|py| -> Result<NodeOutcome<Value, Value>, String> {
                    let callback_bound = callback_arc.as_ref().bind(py);
                    let arg_json = serde_json::to_string(&arg)
                        .map_err(|e| format!("serialize arg failed: {e}"))?;
                    let state_json = serde_json::to_string(&state_snapshot)
                        .map_err(|e| format!("serialize state failed: {e}"))?;
                    let arg_obj = json_string_to_py_obj(py, &arg_json)
                        .map_err(|e| format!("decode arg failed: {e}"))?;
                    let state_obj = json_string_to_py_obj(py, &state_json)
                        .map_err(|e| format!("decode state failed: {e}"))?;
                    let payload_obj = callback_bound
                        .call1((node.as_str(), arg_obj, state_obj))
                        .map_err(|e| format!("callback failed for `{node}`: {e}"))?;
                    let payload_json = py_obj_to_json_string(py, &payload_obj)
                        .map_err(|e| format!("serialize callback payload failed: {e}"))?;
                    let payload_value: Value = serde_json::from_str(&payload_json)
                        .map_err(|e| format!("decode callback payload failed: {e}"))?;
                    let envelope = serde_json::json!({
                        "ok": true,
                        "payload": {
                            "update": payload_value
                                .get("update")
                                .cloned()
                                .unwrap_or(Value::Null),
                            "sends": payload_value
                                .get("sends")
                                .cloned()
                                .unwrap_or(Value::Array(vec![])),
                        },
                        "suspend": payload_value.get("suspend").cloned(),
                    });
                    parse_callback_envelope_json(&envelope.to_string(), &node)
                })
            },
        ));
        self.inner.close_stream();

        let out = run_result.map_err(PyValueError::new_err)?;
        let out_json = serde_json::to_string(&out)
            .map_err(|e| PyValueError::new_err(format!("Serialize state failed: {e}")))?;
        json_string_to_py_obj(py, &out_json)
    }
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
