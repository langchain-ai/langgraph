use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::future::Future;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc as tokio_mpsc, Mutex as AsyncMutex, Notify};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum WaitCondition {
    #[serde(rename = "channel")]
    Channel {
        channel: String,
        min: usize,
        #[serde(default)]
        max: usize,
    },
    #[serde(rename = "timer")]
    Timer { seconds: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnyOfCondition {
    pub conditions: Vec<WaitCondition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "condition")]
pub enum WaitEvent {
    #[serde(rename = "channel")]
    Channel {
        channel: String,
        value: serde_json::Value,
    },
    #[serde(rename = "timer")]
    Timer { seconds: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum WaitRequest {
    #[serde(rename = "condition")]
    Condition { condition: WaitCondition },
    #[serde(rename = "any_of")]
    AnyOf { any_of: AnyOfCondition },
}

pub struct SendPayload<A> {
    pub node: String,
    pub arg: A,
}

pub struct NodeExecResult<U, A> {
    pub update: Option<U>,
    pub sends: Vec<SendPayload<A>>,
}

pub enum NodeOutcome<U, A> {
    Completed(NodeExecResult<U, A>),
    Suspended { wait: WaitRequest },
}

type Task = Box<dyn FnOnce() + Send + 'static>;

fn debug_enabled() -> bool {
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| {
        matches!(
            env::var("DEBUG")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn debug_log(message: &str) {
    if debug_enabled() {
        let current = thread::current();
        let thread_name = current.name().unwrap_or("unnamed");
        println!("[advanced-graph][{thread_name}] {message}");
    }
}

fn pool_size_from_env(var_name: &str, default: usize, min: usize) -> usize {
    let parsed = env::var(var_name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok());
    parsed.unwrap_or(default).max(min)
}

struct ThreadPool {
    tx: mpsc::Sender<Task>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(size: usize, label: &str) -> Self {
        let (tx, rx) = mpsc::channel::<Task>();
        let rx = Arc::new(StdMutex::new(rx));
        let mut workers = Vec::with_capacity(size);
        for idx in 0..size {
            let thread_name = format!("{label}-{idx}");
            let rx = Arc::clone(&rx);
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || loop {
                    let task = {
                        let guard = rx.lock().expect("thread-pool receiver mutex poisoned");
                        guard.recv()
                    };
                    match task {
                        Ok(task) => task(),
                        Err(_) => break,
                    }
                })
                .expect("failed to spawn thread-pool worker");
            workers.push(handle);
        }
        Self {
            tx,
            _workers: workers,
        }
    }

    fn execute<F>(&self, task: F) -> Result<(), String>
    where
        F: FnOnce() + Send + 'static,
    {
        debug_log("thread-pool execute() called");
        self.tx
            .send(Box::new(task))
            .map_err(|e| format!("thread-pool send failed: {e}"))
    }
}

pub fn node_pool_execute<F>(task: F) -> Result<(), String>
where
    F: FnOnce() + Send + 'static,
{
    debug_log("node_pool_execute() called");
    static NODE_POOL: OnceLock<ThreadPool> = OnceLock::new();
    let pool = NODE_POOL.get_or_init(|| {
        let default_size = thread::available_parallelism()
            .map(|n| n.get().max(2))
            .unwrap_or(4);
        let size = pool_size_from_env("LANGGRAPH_NODE_POOL_SIZE", default_size, 1);
        ThreadPool::new(size, "langgraph-node")
    });
    pool.execute(task)
}

pub fn run_loop_pool_execute<F>(task: F) -> Result<(), String>
where
    F: FnOnce() + Send + 'static,
{
    debug_log("run_loop_pool_execute() called");
    run_runtime().spawn_blocking(task);
    Ok(())
}

pub fn run_loop_spawn<F>(future: F) -> Result<(), String>
where
    F: Future<Output = ()> + Send + 'static,
{
    run_runtime().spawn(future);
    Ok(())
}

pub fn run_loop_block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    run_runtime().block_on(future)
}

fn run_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        let default_size = thread::available_parallelism()
            .map(|n| n.get().max(2))
            .unwrap_or(2);
        let worker_threads = pool_size_from_env("LANGGRAPH_RUN_POOL_SIZE", default_size, 1);
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_threads)
            .thread_name("langgraph-runloop")
            .enable_all()
            .build()
            .expect("failed to build tokio runtime")
    })
}

pub fn run_scheduler_loop<U: Send + 'static, A: Send + 'static, FSpawn, FMerge>(
    entry_point: String,
    finish_point: &str,
    initial_arg: A,
    mut spawn: FSpawn,
    mut merge: FMerge,
    rx: mpsc::Receiver<Result<(String, NodeExecResult<U, A>), String>>,
) -> Result<(), String>
where
    FSpawn: FnMut(String, A) -> Result<(), String>,
    FMerge: FnMut(&str, Option<U>) -> Result<(), String>,
{
    debug_log("run_scheduler_loop() started");
    let mut active: usize = 1;
    debug_log("scheduling initial entry node");
    spawn(entry_point, initial_arg)?;

    while active > 0 {
        debug_log(&format!(
            "scheduler waiting for node result (active={active})"
        ));
        let item = rx
            .recv()
            .map_err(|e| format!("scheduler recv failed: {e}"))?;
        active = active.saturating_sub(1);
        let (node_name, node_result) = item.map_err(|e| format!("node execution failed: {e}"))?;
        debug_log(&format!("scheduler received result from node={node_name}"));

        merge(&node_name, node_result.update)?;
        debug_log(&format!("merged update from node={node_name}"));

        if node_name == finish_point {
            debug_log("finish node reached, stopping scheduler loop");
            break;
        }

        for send in node_result.sends {
            active += 1;
            debug_log(&format!(
                "scheduling next node={} (active={active})",
                send.node
            ));
            spawn(send.node, send.arg)?;
        }
    }

    debug_log("run_scheduler_loop() finished");
    Ok(())
}

pub fn merge_json_update(state: &mut Value, update: Option<Value>) {
    debug_log("merge_json_update() called");
    let Some(update_value) = update else {
        debug_log("merge_json_update(): no update payload");
        return;
    };
    match (&mut *state, update_value) {
        (Value::Object(state_obj), Value::Object(update_obj)) => {
            debug_log(&format!(
                "merge_json_update(): object merge with {} keys",
                update_obj.len()
            ));
            for (k, v) in update_obj {
                state_obj.insert(k, v);
            }
        }
        (Value::Object(state_obj), Value::Array(entries)) => {
            debug_log(&format!(
                "merge_json_update(): tuple-list merge with {} entries",
                entries.len()
            ));
            for entry in entries {
                if let Value::Array(pair) = entry {
                    if pair.len() == 2 {
                        if let Value::String(key) = &pair[0] {
                            state_obj.insert(key.clone(), pair[1].clone());
                        }
                    }
                }
            }
        }
        _ => debug_log("merge_json_update(): unsupported update shape, ignored"),
    }
}

#[derive(Clone, Default)]
pub struct Engine {
    channels: Arc<StdMutex<HashMap<String, VecDeque<serde_json::Value>>>>,
    channel_notify: Arc<Notify>,
    stream: Arc<StdMutex<Option<StreamChannel>>>,
}

#[derive(Clone)]
struct StreamChannel {
    sender: tokio_mpsc::UnboundedSender<serde_json::Value>,
    receiver: Arc<AsyncMutex<tokio_mpsc::UnboundedReceiver<serde_json::Value>>>,
}

impl Engine {
    pub fn new() -> Self {
        debug_log("Engine::new()");
        Self::default()
    }

    pub fn add_async_channel(&self, name: &str) {
        debug_log(&format!("Engine::add_async_channel(name={name})"));
        let mut channels = self.channels.lock().expect("channels mutex poisoned");
        channels.entry(name.to_owned()).or_default();
    }

    pub fn start_stream(&self, stream_mode: Option<&str>) -> Result<(), String> {
        let mode = stream_mode.and_then(|m| {
            let trimmed = m.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        });
        let mut stream = self.stream.lock().expect("stream mutex poisoned");
        if let Some(mode) = mode {
            if mode != "custom" {
                return Err(format!(
                    "unsupported stream_mode `{mode}`, only `custom` is supported"
                ));
            }
            let (sender, receiver) = tokio_mpsc::unbounded_channel();
            *stream = Some(StreamChannel {
                sender,
                receiver: Arc::new(AsyncMutex::new(receiver)),
            });
        } else {
            *stream = None;
        }
        Ok(())
    }

    pub fn close_stream(&self) {
        let mut stream = self.stream.lock().expect("stream mutex poisoned");
        *stream = None;
    }

    pub fn send_custom_stream_event(&self, value: serde_json::Value) {
        let sender = {
            let stream = self.stream.lock().expect("stream mutex poisoned");
            stream.as_ref().map(|s| s.sender.clone())
        };
        if let Some(tx) = sender {
            let _ = tx.send(value);
        }
    }

    pub async fn receive_stream_async(&self) -> Option<serde_json::Value> {
        let receiver = {
            let stream = self.stream.lock().expect("stream mutex poisoned");
            stream.as_ref().map(|s| Arc::clone(&s.receiver))
        };
        let Some(receiver) = receiver else {
            return None;
        };
        let mut guard = receiver.lock().await;
        guard.recv().await
    }

    pub fn publish_json(&self, channel: &str, value: serde_json::Value) -> Result<(), String> {
        debug_log(&format!("Engine::publish_json(channel={channel})"));
        let mut channels = self.channels.lock().expect("channels mutex poisoned");
        let queue = channels
            .get_mut(channel)
            .ok_or_else(|| format!("Unknown channel `{channel}`"))?;
        queue.push_back(value);
        self.channel_notify.notify_waiters();
        Ok(())
    }

    pub async fn wait_request_async(&self, wait: &WaitRequest) -> Result<WaitEvent, String> {
        match wait {
            WaitRequest::Condition { condition } => self.wait_for_async(condition).await,
            WaitRequest::AnyOf { any_of } => self.wait_for_any_of_async(any_of).await,
        }
    }

    pub async fn wait_for_async(&self, cond: &WaitCondition) -> Result<WaitEvent, String> {
        debug_log(&format!("Engine::wait_for_async(cond={cond:?})"));
        match cond {
            WaitCondition::Channel { channel, min, max } => {
                if *min < 1 {
                    return Err("channel condition min must be >= 1".to_string());
                }
                if *max != 0 && *max < *min {
                    return Err("channel condition max must be 0 or >= min".to_string());
                }
                loop {
                    if let Some(event) = self.try_take_channel_event(channel, *min, *max)? {
                        return Ok(event);
                    }
                    self.channel_notify.notified().await;
                }
            }
            WaitCondition::Timer { seconds } => {
                if *seconds <= 0.0 {
                    return Err("timer condition must be > 0".to_string());
                }
                tokio::time::sleep(Duration::from_secs_f64(*seconds)).await;
                Ok(WaitEvent::Timer { seconds: *seconds })
            }
        }
    }

    pub async fn wait_for_any_of_async(
        &self,
        any_of: &AnyOfCondition,
    ) -> Result<WaitEvent, String> {
        debug_log(&format!(
            "Engine::wait_for_any_of_async(conditions={})",
            any_of.conditions.len()
        ));
        if any_of.conditions.is_empty() {
            return Err("any_of requires at least one condition".to_string());
        }

        let started = Instant::now();
        let mut min_timer: Option<f64> = None;
        for cond in &any_of.conditions {
            if let WaitCondition::Timer { seconds } = cond {
                if *seconds <= 0.0 {
                    return Err("timer condition must be > 0".to_string());
                }
                min_timer = Some(min_timer.map_or(*seconds, |x| x.min(*seconds)));
            }
        }

        loop {
            if let Some(event) = self.try_take_any_of_channel_events(any_of)? {
                return Ok(event);
            }

            if let Some(seconds) = min_timer {
                let timeout = Duration::from_secs_f64(seconds);
                let elapsed = started.elapsed();
                if elapsed >= timeout {
                    return Ok(WaitEvent::Timer { seconds });
                }
                let remaining = timeout.saturating_sub(elapsed);
                tokio::select! {
                    _ = self.channel_notify.notified() => {}
                    _ = tokio::time::sleep(remaining) => {
                        return Ok(WaitEvent::Timer { seconds });
                    }
                }
            } else {
                self.channel_notify.notified().await;
            }
        }
    }

    pub fn wait_for(&self, cond: &WaitCondition) -> Result<WaitEvent, String> {
        run_loop_block_on(self.wait_for_async(cond))
    }

    pub fn wait_for_any_of(&self, any_of: &AnyOfCondition) -> Result<WaitEvent, String> {
        run_loop_block_on(self.wait_for_any_of_async(any_of))
    }

    fn try_take_channel_event(
        &self,
        channel: &str,
        min: usize,
        max: usize,
    ) -> Result<Option<WaitEvent>, String> {
        let mut channels = self.channels.lock().expect("channels mutex poisoned");
        let queue = channels
            .get_mut(channel)
            .ok_or_else(|| format!("Unknown channel `{channel}`"))?;
        if queue.len() < min {
            return Ok(None);
        }
        let take_count = if max == 0 { min } else { queue.len().min(max) };
        if take_count == 1 {
            if let Some(value) = queue.pop_front() {
                return Ok(Some(WaitEvent::Channel {
                    channel: channel.to_string(),
                    value,
                }));
            }
            return Ok(None);
        }
        let mut values = Vec::with_capacity(take_count);
        for _ in 0..take_count {
            if let Some(v) = queue.pop_front() {
                values.push(v);
            }
        }
        Ok(Some(WaitEvent::Channel {
            channel: channel.to_string(),
            value: serde_json::Value::Array(values),
        }))
    }

    fn try_take_any_of_channel_events(
        &self,
        any_of: &AnyOfCondition,
    ) -> Result<Option<WaitEvent>, String> {
        let mut channels = self.channels.lock().expect("channels mutex poisoned");
        let mut consumed_per_channel: HashMap<String, usize> = HashMap::new();
        let mut plans: Vec<(String, usize)> = Vec::new();

        for cond in &any_of.conditions {
            let WaitCondition::Channel { channel, min, max } = cond else {
                continue;
            };
            if *min < 1 {
                return Err("channel condition min must be >= 1".to_string());
            }
            if *max != 0 && *max < *min {
                return Err("channel condition max must be 0 or >= min".to_string());
            }

            let queue = channels
                .get(channel)
                .ok_or_else(|| format!("Unknown channel `{channel}`"))?;
            let already_planned = consumed_per_channel.get(channel).copied().unwrap_or(0);
            let available = queue.len().saturating_sub(already_planned);
            if available < *min {
                continue;
            }

            let take_count = if *max == 0 { *min } else { available.min(*max) };
            plans.push((channel.clone(), take_count));
            consumed_per_channel
                .entry(channel.clone())
                .and_modify(|v| *v += take_count)
                .or_insert(take_count);
        }

        if plans.is_empty() {
            return Ok(None);
        }

        if plans.len() == 1 {
            let (channel, take_count) = &plans[0];
            let queue = channels
                .get_mut(channel)
                .ok_or_else(|| format!("Unknown channel `{channel}`"))?;
            let value = pop_queue_value(queue, *take_count);
            return Ok(Some(WaitEvent::Channel {
                channel: channel.clone(),
                value,
            }));
        }

        let mut matched = Vec::with_capacity(plans.len());
        for (channel, take_count) in plans {
            let queue = channels
                .get_mut(&channel)
                .ok_or_else(|| format!("Unknown channel `{channel}`"))?;
            let value = pop_queue_value(queue, take_count);
            matched.push(json!({
                "channel": channel,
                "value": value,
            }));
        }

        Ok(Some(WaitEvent::Channel {
            channel: "__any_of__".to_string(),
            value: Value::Array(matched),
        }))
    }
}

fn pop_queue_value(queue: &mut VecDeque<Value>, take_count: usize) -> Value {
    if take_count == 1 {
        return queue.pop_front().unwrap_or(Value::Null);
    }
    let mut values = Vec::with_capacity(take_count);
    for _ in 0..take_count {
        if let Some(v) = queue.pop_front() {
            values.push(v);
        }
    }
    Value::Array(values)
}

#[derive(Debug, Deserialize)]
struct CallbackSendPayloadJson {
    node: String,
    #[serde(default)]
    arg: Value,
}

#[derive(Debug, Deserialize)]
struct CallbackNodeExecResultJsonWire {
    update: Option<Value>,
    #[serde(default)]
    sends: Vec<CallbackSendPayloadJson>,
}

#[derive(Debug, Deserialize)]
struct CallbackEnvelopeIn {
    ok: bool,
    #[serde(default)]
    payload: Option<CallbackNodeExecResultJsonWire>,
    #[serde(default)]
    suspend: Option<WaitRequest>,
    #[serde(default)]
    error: Option<String>,
}

enum SchedulerEvent<U, A> {
    Node(Result<NodeExecution<U, A>, String>),
    Resume {
        node: String,
        arg: A,
        event: WaitEvent,
    },
    WaitError(String),
}

struct NodeExecution<U, A> {
    node: String,
    arg: A,
    outcome: NodeOutcome<U, A>,
}

fn spawn_node_task<State, U, A, F>(
    node: String,
    arg: A,
    state_snapshot: State,
    tx: tokio_mpsc::UnboundedSender<SchedulerEvent<U, A>>,
    callback: Arc<F>,
) -> Result<(), String>
where
    State: Send + 'static,
    U: Send + 'static,
    A: Clone + Send + 'static,
    F: Fn(String, A, State) -> Result<NodeOutcome<U, A>, String> + Send + Sync + 'static,
{
    node_pool_execute(move || {
        let node_for_result = node.clone();
        let arg_for_result = arg.clone();
        let result = callback(node, arg, state_snapshot).map(|outcome| NodeExecution {
            node: node_for_result,
            arg: arg_for_result,
            outcome,
        });
        let _ = tx.send(SchedulerEvent::Node(result));
    })
}

pub async fn run_graph_with_callback<State, U, A, FCallback, FMerge, FWrap>(
    entry_point: String,
    finish_point: String,
    initial_state: State,
    initial_input: A,
    engine: Engine,
    callback: FCallback,
    merge_update: FMerge,
    wrap_resume_arg: FWrap,
) -> Result<State, String>
where
    State: Clone + Send + 'static,
    U: Send + 'static,
    A: Clone + Send + 'static,
    FCallback: Fn(String, A, State) -> Result<NodeOutcome<U, A>, String> + Send + Sync + 'static,
    FMerge: Fn(&mut State, Option<U>) -> Result<(), String> + Send + Sync + 'static,
    FWrap: Fn(A, WaitEvent) -> Result<A, String> + Send + Sync + 'static,
{
    let callback = Arc::new(callback);
    let merge_update = Arc::new(merge_update);
    let wrap_resume_arg = Arc::new(wrap_resume_arg);

    let (tx, mut rx) = tokio_mpsc::unbounded_channel::<SchedulerEvent<U, A>>();
    let state = Arc::new(StdMutex::new(initial_state));
    let tx_for_spawn = tx.clone();
    let state_for_spawn = Arc::clone(&state);
    let mut active: usize = 1;
    let mut waiting: usize = 0;

    spawn_node_task(
        entry_point,
        initial_input,
        state_for_spawn
            .lock()
            .expect("state mutex poisoned")
            .clone(),
        tx_for_spawn.clone(),
        Arc::clone(&callback),
    )?;

    while active > 0 || waiting > 0 {
        let evt = rx
            .recv()
            .await
            .ok_or_else(|| "scheduler event channel closed".to_string())?;
        match evt {
            SchedulerEvent::Node(result) => {
                active = active.saturating_sub(1);
                let exec = result?;
                match exec.outcome {
                    NodeOutcome::Completed(node_result) => {
                        let mut guard = state.lock().expect("state mutex poisoned");
                        merge_update(&mut guard, node_result.update)?;
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
                            spawn_node_task(
                                send.node,
                                send.arg,
                                snapshot,
                                tx_for_spawn.clone(),
                                Arc::clone(&callback),
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
                                    let _ =
                                        tx_wait.send(SchedulerEvent::Resume { node, arg, event });
                                }
                                Err(e) => {
                                    let _ = tx_wait.send(SchedulerEvent::WaitError(e));
                                }
                            }
                        });
                    }
                }
            }
            SchedulerEvent::Resume { node, arg, event } => {
                waiting = waiting.saturating_sub(1);
                active += 1;
                let snapshot = state_for_spawn
                    .lock()
                    .expect("state mutex poisoned")
                    .clone();
                let resume_arg = wrap_resume_arg(arg, event)?;
                spawn_node_task(
                    node,
                    resume_arg,
                    snapshot,
                    tx_for_spawn.clone(),
                    Arc::clone(&callback),
                )?;
            }
            SchedulerEvent::WaitError(e) => return Err(e),
        }
    }

    let final_state = state.lock().expect("state mutex poisoned").clone();
    Ok(final_state)
}

pub async fn run_graph_json_with_callback<F>(
    entry_point: String,
    finish_point: String,
    initial_state: Value,
    initial_input: Value,
    engine: Engine,
    callback: F,
) -> Result<Value, String>
where
    F: Fn(String, Value, Value) -> Result<NodeOutcome<Value, Value>, String>
        + Send
        + Sync
        + 'static,
{
    run_graph_with_callback(
        entry_point,
        finish_point,
        initial_state,
        initial_input,
        engine,
        callback,
        |state: &mut Value, update: Option<Value>| {
            merge_json_update(state, update);
            Ok(())
        },
        |arg: Value, event: WaitEvent| {
            Ok(serde_json::json!({
                "__lg_resume_arg__": arg,
                "__lg_resume_event__": event,
            }))
        },
    )
    .await
}

pub fn parse_callback_envelope_json(
    raw: &str,
    node_name: &str,
) -> Result<NodeOutcome<Value, Value>, String> {
    let parsed: CallbackEnvelopeIn = serde_json::from_str(raw)
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
