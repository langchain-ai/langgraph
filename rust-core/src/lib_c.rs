use crate::engine::{
    merge_json_update, node_pool_execute, run_loop_block_on, run_loop_spawn, AnyOfCondition,
    Engine, NodeExecResult, NodeOutcome, SendPayload, WaitEvent, WaitRequest,
};
use serde::Deserialize;
use serde_json::Value;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc as tokio_mpsc;

#[derive(Debug, Deserialize)]
struct SendPayloadJson {
    node: String,
    #[serde(default)]
    arg: Value,
}

#[derive(Debug, Deserialize)]
struct NodeExecResultJsonWire {
    update: Option<Value>,
    #[serde(default)]
    sends: Vec<SendPayloadJson>,
}

#[derive(Debug, Deserialize)]
struct CallbackEnvelopeIn {
    ok: bool,
    #[serde(default)]
    payload: Option<NodeExecResultJsonWire>,
    #[serde(default)]
    suspend: Option<WaitRequest>,
    #[serde(default)]
    error: Option<String>,
}

type CNodeCallback = unsafe extern "C" fn(
    user_data: libc::c_ulong,
    node: *mut c_char,
    arg_json: *mut c_char,
    state_json: *mut c_char,
) -> *mut c_char;

#[derive(Clone, Copy)]
struct CUserData(libc::c_ulong);

fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("Received null pointer".to_string());
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str()
        .map_err(|e| format!("Invalid UTF-8 input string: {e}"))
}

fn into_c_ptr(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => CString::new("{\"error\":\"NUL byte in output\"}")
            .expect("static string is valid")
            .into_raw(),
    }
}

fn parse_c_callback_result(
    raw: String,
    node_name: &str,
) -> Result<NodeOutcome<Value, Value>, String> {
    let parsed: CallbackEnvelopeIn = serde_json::from_str(&raw)
        .map_err(|e| format!("decode callback envelope for `{node_name}` failed: {e}"))?;
    if !parsed.ok {
        return Err(parsed
            .error
            .unwrap_or_else(|| format!("callback reported error for `{node_name}`")));
    }
    if let Some(wait) = parsed.suspend {
        return Ok(NodeOutcome::Suspended { wait });
    }
    let payload = parsed
        .payload
        .ok_or_else(|| format!("callback payload missing for `{node_name}`"))?;
    let sends = payload
        .sends
        .into_iter()
        .map(|s| SendPayload {
            node: s.node,
            arg: s.arg,
        })
        .collect();
    Ok(NodeOutcome::Completed(NodeExecResult {
        update: payload.update,
        sends,
    }))
}

enum SchedulerEventJson {
    Node(Result<NodeExecutionJson, String>),
    Resume {
        node: String,
        arg: Value,
        event: WaitEvent,
    },
    WaitError(String),
}

struct NodeExecutionJson {
    node: String,
    arg: Value,
    outcome: NodeOutcome<Value, Value>,
}

fn spawn_json_node_task(
    node: String,
    arg: Value,
    state_snapshot: Value,
    tx: tokio_mpsc::UnboundedSender<SchedulerEventJson>,
    user_data_bits: libc::c_ulong,
    callback: CNodeCallback,
) -> Result<(), String> {
    node_pool_execute(move || {
        let node_for_result = node.clone();
        let arg_for_result = arg.clone();
        let result = (|| -> Result<NodeExecutionJson, String> {
            let node_c =
                CString::new(node.clone()).map_err(|e| format!("invalid node name: {e}"))?;
            let arg_json = serde_json::to_string(&arg)
                .map_err(|e| format!("serialize arg for `{node}` failed: {e}"))?;
            let state_json = serde_json::to_string(&state_snapshot)
                .map_err(|e| format!("serialize state for `{node}` failed: {e}"))?;
            let arg_c =
                CString::new(arg_json).map_err(|e| format!("invalid arg JSON bytes: {e}"))?;
            let state_c =
                CString::new(state_json).map_err(|e| format!("invalid state JSON bytes: {e}"))?;
            let out_ptr = unsafe {
                callback(
                    user_data_bits,
                    node_c.as_ptr() as *mut c_char,
                    arg_c.as_ptr() as *mut c_char,
                    state_c.as_ptr() as *mut c_char,
                )
            };
            if out_ptr.is_null() {
                return Err(format!("callback returned null for `{node}`"));
            }
            let out_raw = unsafe { CStr::from_ptr(out_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe {
                libc::free(out_ptr.cast());
            }
            let payload = parse_c_callback_result(out_raw, &node)?;
            Ok(NodeExecutionJson {
                node: node_for_result,
                arg: arg_for_result,
                outcome: payload,
            })
        })();
        let _ = tx.send(SchedulerEventJson::Node(result));
    })
}

async fn run_graph_scheduler_json(
    entry_point: String,
    finish_point: String,
    initial_state: Value,
    initial_input: Value,
    engine: Engine,
    user_data: CUserData,
    callback: CNodeCallback,
) -> Result<Value, String> {
    let (tx, mut rx) = tokio_mpsc::unbounded_channel::<SchedulerEventJson>();
    let state = Arc::new(Mutex::new(initial_state));
    let user_data_bits = user_data.0;
    let tx_for_spawn = tx.clone();
    let state_for_spawn = Arc::clone(&state);
    let state_for_merge = Arc::clone(&state);
    let mut active: usize = 1;
    let mut waiting: usize = 0;
    spawn_json_node_task(
        entry_point,
        initial_input,
        state_for_spawn
            .lock()
            .expect("state mutex poisoned")
            .clone(),
        tx_for_spawn.clone(),
        user_data_bits,
        callback,
    )?;
    while active > 0 || waiting > 0 {
        let evt = rx
            .recv()
            .await
            .ok_or_else(|| "scheduler event channel closed".to_string())?;
        match evt {
            SchedulerEventJson::Node(result) => {
                active = active.saturating_sub(1);
                let exec = result?;
                match exec.outcome {
                    NodeOutcome::Completed(node_result) => {
                        let mut guard = state_for_merge.lock().expect("state mutex poisoned");
                        merge_json_update(&mut guard, node_result.update);
                        drop(guard);
                        if exec.node == finish_point {
                            break;
                        }
                        for send in node_result.sends {
                            active += 1;
                            let snapshot = state_for_spawn
                                .lock()
                                .expect("state mutex poisoned")
                                .clone();
                            spawn_json_node_task(
                                send.node,
                                send.arg,
                                snapshot,
                                tx_for_spawn.clone(),
                                user_data_bits,
                                callback,
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
                            match engine_for_wait.wait_request_async(&wait).await {
                                Ok(event) => {
                                    let _ = tx_wait.send(SchedulerEventJson::Resume { node, arg, event });
                                }
                                Err(e) => {
                                    let _ = tx_wait.send(SchedulerEventJson::WaitError(e));
                                }
                            }
                        });
                    }
                }
            }
            SchedulerEventJson::Resume { node, arg, event } => {
                waiting = waiting.saturating_sub(1);
                active += 1;
                let snapshot = state_for_spawn
                    .lock()
                    .expect("state mutex poisoned")
                    .clone();
                spawn_json_node_task(
                    node,
                    wrap_resume_arg(arg, event),
                    snapshot,
                    tx_for_spawn.clone(),
                    user_data_bits,
                    callback,
                )?;
            }
            SchedulerEventJson::WaitError(e) => return Err(e),
        }
    }
    let final_state = state.lock().expect("state mutex poisoned").clone();
    Ok(final_state)
}

fn wrap_resume_arg(arg: Value, event: WaitEvent) -> Value {
    serde_json::json!({
        "__lg_resume_arg__": arg,
        "__lg_resume_event__": event,
    })
}

#[no_mangle]
pub extern "C" fn rc_engine_new() -> *mut Engine {
    Box::into_raw(Box::new(Engine::new()))
}

#[no_mangle]
/// # Safety
/// `ptr` must be either null or a valid pointer returned by `rc_engine_new`.
pub unsafe extern "C" fn rc_engine_free(ptr: *mut Engine) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr));
}

#[no_mangle]
/// # Safety
/// `ptr` must be either null or a valid pointer returned by this library.
pub unsafe extern "C" fn rc_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    drop(CString::from_raw(ptr));
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `channel` must be a valid null-terminated UTF-8 string pointer.
pub unsafe extern "C" fn rc_add_async_channel(
    ptr: *mut Engine,
    channel: *const c_char,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let channel = match cstr_to_str(channel) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    (*ptr).add_async_channel(channel);
    into_c_ptr("{\"ok\":true}".to_string())
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `channel` and `value_json` must be valid null-terminated UTF-8 string pointers.
pub unsafe extern "C" fn rc_publish_json(
    ptr: *mut Engine,
    channel: *const c_char,
    value_json: *const c_char,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let channel = match cstr_to_str(channel) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let value_json = match cstr_to_str(value_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let value: Value = match serde_json::from_str(value_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid JSON value: {e}\"}}"
            ))
        }
    };
    let result = (*ptr).publish_json(channel, value);
    match result {
        Ok(()) => into_c_ptr("{\"ok\":true}".to_string()),
        Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    }
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `any_of_json` must be a valid null-terminated UTF-8 string pointer.
pub unsafe extern "C" fn rc_wait_any_of_json(
    ptr: *mut Engine,
    any_of_json: *const c_char,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let any_of_json = match cstr_to_str(any_of_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let any_of: AnyOfCondition = match serde_json::from_str(any_of_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid any_of JSON: {e}\"}}"
            ))
        }
    };
    let result = run_loop_block_on((*ptr).wait_for_any_of_async(&any_of));
    match result {
        Ok(event) => match serde_json::to_string(&event) {
            Ok(s) => into_c_ptr(format!("{{\"ok\":true,\"event\":{s}}}")),
            Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
        },
        Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    }
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `entry_point`, `finish_point`, and `initial_state_json` must be valid null-terminated UTF-8 pointers.
/// `callback` must be a valid function pointer that returns a malloc-allocated C string.
pub unsafe extern "C" fn rc_run_graph_json(
    ptr: *mut Engine,
    entry_point: *const c_char,
    finish_point: *const c_char,
    initial_state_json: *const c_char,
    initial_input_json: *const c_char,
    user_data: libc::c_ulong,
    callback: Option<CNodeCallback>,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let Some(callback) = callback else {
        return into_c_ptr("{\"ok\":false,\"error\":\"null callback pointer\"}".to_string());
    };
    let entry_point = match cstr_to_str(entry_point) {
        Ok(v) => v.to_string(),
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let finish_point = match cstr_to_str(finish_point) {
        Ok(v) => v.to_string(),
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let initial_state_json = match cstr_to_str(initial_state_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let initial_state: Value = match serde_json::from_str(initial_state_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid initial_state JSON: {e}\"}}"
            ))
        }
    };
    let initial_input_json = match cstr_to_str(initial_input_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let initial_input: Value = match serde_json::from_str(initial_input_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid initial_input JSON: {e}\"}}"
            ))
        }
    };

    let (tx, rx) = mpsc::channel::<Result<Value, String>>();
    let user_data = CUserData(user_data);
    let run_engine = (*ptr).clone();
    let submit = run_loop_spawn(async move {
        let out = run_graph_scheduler_json(
            entry_point,
            finish_point,
            initial_state,
            initial_input,
            run_engine,
            user_data,
            callback,
        )
        .await;
        let _ = tx.send(out);
    });
    if let Err(e) = submit {
        return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}"));
    }

    let run_result = match rx.recv() {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"run-loop recv failed: {e}\"}}"
            ))
        }
    };
    match run_result {
        Ok(state) => match serde_json::to_string(&state) {
            Ok(s) => into_c_ptr(format!("{{\"ok\":true,\"state\":{s}}}")),
            Err(e) => into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"serialize state failed: {e}\"}}"
            )),
        },
        Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    }
}
