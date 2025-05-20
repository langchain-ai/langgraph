package pregel

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

func PrepareNextTasks(
	ctx context.Context,
	checkpoint Checkpoint,
	pendingWrites []interface{},
	processes map[string]PregelNode,
	channels map[string]BaseChannel,
	managed ManagedValueMapping,
	config RunnableConfig,
	step int,
	forExecution bool,
	store BaseStore,
	checkpointer BaseCheckpointSaver,
	// ─ optimisation hints (optional) ─
	triggerToNodes map[string][]string,
	updatedChannels map[string]struct{},
) (map[string]interface{}, error) {

	// Decode checkpoint.id (UUID/xxhash) into raw bytes for deterministic task-id hashing.
	cleanID := strings.ReplaceAll(checkpoint.ID, "-", "")
	checkpointIDBytes, err := hex.DecodeString(cleanID)
	if err != nil {
		return nil, err
	}

	nullVersion := checkpointNullVersion(checkpoint)
	tasks := make(map[string]interface{})

	// Consume pending sends
	for idx := range checkpoint.PendingSends {
		task, err := PrepareSingleTask(
			ctx,
			[]interface{}{PUSH, idx},
			"",
			checkpoint,
			checkpointIDBytes,
			nullVersion,
			pendingWrites,
			processes,
			channels,
			managed,
			config,
			step,
			forExecution,
			store,
			checkpointer,
		)
		if err != nil {
			return nil, err
		}
		if task == nil {
			continue
		}
		if id, ok := taskID(task); ok {
			tasks[id] = task
		}
	}

	var candidateNodes []string

	if len(updatedChannels) > 0 && len(triggerToNodes) > 0 {
		nodeSet := map[string]struct{}{}
		for ch := range updatedChannels {
			for _, n := range triggerToNodes[ch] {
				nodeSet[n] = struct{}{}
			}
		}
		for n := range nodeSet {
			candidateNodes = append(candidateNodes, n)
		}
		sort.Strings(candidateNodes) // deterministic order
	} else if len(checkpoint.ChannelVersions) == 0 {
		candidateNodes = nil
	} else {
		for n := range processes {
			candidateNodes = append(candidateNodes, n)
		}
		sort.Strings(candidateNodes)
	}

	for _, name := range candidateNodes {
		task, err := PrepareSingleTask(
			ctx,
			[]interface{}{PULL, name},
			"", // checksum only used when resuming a partial step
			checkpoint,
			checkpointIDBytes,
			nullVersion,
			pendingWrites,
			processes,
			channels,
			managed,
			config,
			step,
			forExecution,
			store,
			checkpointer,
		)
		if err != nil {
			return nil, err
		}
		if task == nil {
			continue
		}
		if id, ok := taskID(task); ok {
			tasks[id] = task
		}
	}

	return tasks, nil
}

func PrepareSingleTask(
	ctx context.Context,
	taskPath []interface{}, // e.g. [PUSH, idx]  OR  [PULL, "node"]
	taskIDChecksum string, // optional – used when resuming
	checkpoint Checkpoint, // state captured at end of previous step
	checkpointIDBytes []byte, // checkpoint.id as bytes (uuid / xxhash)
	checkpointNullVersion interface{}, // sentinel “null” version value
	pendingWrites []interface{}, // successful writes from *this* step so far
	processes map[string]PregelNode, // graph definition
	channels map[string]BaseChannel, // live channel values
	managed ManagedValueMapping, // placeholder resolver
	config RunnableConfig, // config inherited from graph.Invoke()
	step int, // current super-step (n+1)
	forExecution bool, // false = planning pass, true = exec pass
	store BaseStore, // needed for reads/writes
	checkpointer BaseCheckpointSaver, // used only when executing
) (interface{}, error) {
	// Ensure checkpoint.ChannelVersions is initialized
	if checkpoint.ChannelVersions == nil {
		checkpoint.ChannelVersions = make(map[string]int64)
	}

	cfgSection := config.Configurable
	if cfgSection == nil {
		cfgSection = map[string]interface{}{}
	}
	parentNS, _ := cfgSection[CONFIG_KEY_CHECKPOINT_NS].(string)

	emitConfig := func(base RunnableConfig, md map[string]interface{}) RunnableConfig {
		// Make a shallow copy of the struct
		out := base

		if out.Configurable == nil {
			out.Configurable = map[string]interface{}{}
		}
		confClone := make(map[string]interface{}, len(out.Configurable))
		for k, v := range out.Configurable {
			confClone[k] = v
		}
		confClone[CONFIG_KEY_SCRATCHPAD] = createScratchpad(
			out.Configurable[CONFIG_KEY_SCRATCHPAD].(map[string]interface{}),
			pendingWrites,
			md["langgraph_checkpoint_ns"].(string),
			md["langgraph_checkpoint_ns"].(string),
			out.Configurable[CONFIG_KEY_RESUME_MAP].(map[string]interface{}),
		)
		confClone[CONFIG_KEY_CHECKPOINTER] = checkpointer
		out.Configurable = confClone

		if out.Metadata == nil {
			out.Metadata = map[string]interface{}{}
		}
		for k, v := range md {
			out.Metadata[k] = v
		}

		return out
	}

	// Convenience for checksum comparison
	checkSumMatch := func(need string) error {
		if taskIDChecksum != "" && taskIDChecksum != need {
			return fmt.Errorf("%s != %s", need, taskIDChecksum)
		}
		return nil
	}

	// PUSH
	if len(taskPath) > 0 && taskPath[0] == PUSH {

		// PUSH triggered via explicit Call (happens during node execution)
		//      taskPath shape: [PUSH, parentPath, writeIdx, parentTaskID, Call]
		if len(taskPath) >= 5 {
			call, ok := taskPath[4].(Call)
			if ok {
				name, isStr := call.Func.(string)
				if !isStr {
					name = "unknown"
				}

				// Hash-stable checkpoint namespace
				var checkpointNS string
				if parentNS == "" {
					checkpointNS = name
				} else {
					checkpointNS = parentNS + NS_SEP + name
				}

				// Deterministic task-id
				taskID := taskIDFunc(
					checkpointIDBytes,
					checkpointNS,
					strconv.Itoa(step),
					name,
					PUSH,
					taskPathStr(taskPath[1]),
					fmt.Sprintf("%v", taskPath[2]),
				)
				if err := checkSumMatch(taskID); err != nil {
					return nil, err
				}

				taskCheckpointNS := checkpointNS + NS_END + taskID
				metadata := map[string]interface{}{
					"langgraph_step":          step,
					"langgraph_node":          name,
					"langgraph_triggers":      []string{PUSH},
					"langgraph_path":          taskPath[:3],
					"langgraph_checkpoint_ns": taskCheckpointNS,
				}

				if forExecution {
					var node NodeRunnable
					if proc, ok := processes[name]; ok {
						node = proc.Node
					}

					return PregelExecutableTask{
						PregelTask: PregelTask{
							ID:   taskID,
							Name: name,
							Path: taskPath[:3],
						},
						Input:    call.Input,
						Node:     node,
						Writes:   []Write{},
						Config:   emitConfig(config, metadata),
						Triggers: []string{PUSH},
					}, nil
				}
				return PregelTask{ID: taskID, Name: name, Path: taskPath[:3]}, nil
			}
		}

		// ---------------------------------------------------------------------
		// 1b. Standard pending-send packet: taskPath shape [PUSH, idx]
		// ---------------------------------------------------------------------
		if len(taskPath) == 2 {
			idx, ok := taskPath[1].(int)
			if !ok || idx >= len(checkpoint.PendingSends) {
				return nil, nil
			}

			packet := checkpoint.PendingSends[idx]
			proc, ok := processes[packet.Node]
			if !ok || proc.Node == nil {
				return nil, nil
			}

			checkpointNS := parentNS
			if checkpointNS != "" {
				checkpointNS += NS_SEP + packet.Node
			} else {
				checkpointNS = packet.Node
			}

			taskID := taskIDFunc(
				checkpointIDBytes,
				checkpointNS,
				strconv.Itoa(step),
				packet.Node,
				PUSH,
				strconv.Itoa(idx),
			)
			if err := checkSumMatch(taskID); err != nil {
				return nil, err
			}

			taskCheckpointNS := checkpointNS + NS_END + taskID
			metadata := map[string]interface{}{
				"langgraph_step":          step,
				"langgraph_node":          packet.Node,
				"langgraph_triggers":      []string{PUSH},
				"langgraph_path":          taskPath,
				"langgraph_checkpoint_ns": taskCheckpointNS,
			}

			if forExecution {
				return PregelExecutableTask{
					PregelTask: PregelTask{
						ID:   taskID,
						Name: packet.Node,
						Path: taskPath,
					},
					Input:    packet.Arg,
					Node:     proc.Node,
					Writes:   nil,
					Config:   emitConfig(config, metadata),
					Triggers: []string{PUSH},
				}, nil
			}
			return PregelTask{ID: taskID, Name: packet.Node, Path: taskPath}, nil
		}

		// An ill-formed PUSH path – nothing to schedule
		return nil, nil
	}

	// PULL branch
	if len(taskPath) > 0 && taskPath[0] == PULL {
		if len(taskPath) < 2 {
			return nil, nil
		}
		name, ok := taskPath[1].(string)
		if !ok {
			return nil, nil
		}

		proc, ok := processes[name]
		if !ok || proc.Node == nil {
			return nil, nil
		}

		seen := map[string]interface{}{}
		if v, _ := checkpoint.VersionsSeen[name].(map[string]interface{}); v != nil {
			for k, vv := range v { // shallow copy
				seen[k] = vv
			}
		}

		var triggers []string
		for _, ch := range proc.Triggers {
			cv, exists := checkpoint.ChannelVersions[ch]
			if !exists {
				cv = checkpointNullVersion.(int64) // use the provided null version
			}
			sv, _ := seen[ch].(int64) // default to 0 if not exists or wrong type
			if compareVersion(cv, sv) > 0 {
				triggers = append(triggers, ch)
			}
		}
		if len(triggers) == 0 {
			return nil, nil // not ready
		}
		sort.Strings(triggers)

		input := map[string]interface{}{}
		for _, ch := range proc.Triggers {
			if v, ok := channels[ch]; ok {
				input[ch] = v
			}
		}

		checkpointNS := parentNS
		if checkpointNS != "" {
			checkpointNS += NS_SEP + name
		} else {
			checkpointNS = name
		}

		taskID := taskIDFunc(
			checkpointIDBytes,
			checkpointNS,
			strconv.Itoa(step),
			name,
			PULL,
			// join triggers to guarantee deterministic id
			fmt.Sprintf("%v", triggers),
		)
		if err := checkSumMatch(taskID); err != nil {
			return nil, err
		}

		taskCheckpointNS := checkpointNS + NS_END + taskID
		metadata := map[string]interface{}{
			"langgraph_step":          step,
			"langgraph_node":          name,
			"langgraph_triggers":      triggers,
			"langgraph_path":          taskPath,
			"langgraph_checkpoint_ns": taskCheckpointNS,
		}

		if forExecution {
			return PregelExecutableTask{
				PregelTask: PregelTask{
					ID:   taskID,
					Name: name,
					Path: taskPath,
				},
				Input:    input,
				Node:     proc.Node,
				Writes:   nil,
				Config:   emitConfig(config, metadata),
				Triggers: triggers,
			}, nil
		}
		return PregelTask{ID: taskID, Name: name, Path: taskPath}, nil
	}

	return nil, nil
}

// Private / Helpers

// taskIDFunc deterministically hashes the checkpoint-scoped information that
// must be unique for a task in a given super-step.
func taskIDFunc(checkpointIDBytes []byte, parts ...string) string {
	h := sha256.New()
	_, _ = h.Write(checkpointIDBytes)
	for _, p := range parts {
		_, _ = h.Write([]byte(p))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// taskPathStr is only used so the path element contributes to the hash in a
// deterministic textual form.
func taskPathStr(path interface{}) string {
	return fmt.Sprintf("%v", path)
}

// createScratchpad returns an *immutable* copy of the scratchpad that will
// be injected into the task-local Config.  We:
//
//  1. start from the previous scratchpad (if any),
//  2. merge in any successful writes from earlier tasks in this super-step,
//  3. copy-on-write so individual tasks never share interior maps.
//
// The logic below is intentionally simple; extend as needed.
func createScratchpad(
	current map[string]interface{},
	pendingWrites []interface{},
	taskID string,
	checkpointHash string,
	resumeMap map[string]interface{},
) map[string]interface{} {
	out := map[string]interface{}{}
	for k, v := range current {
		out[k] = v
	}
	if len(pendingWrites) > 0 {
		out["pending_writes"] = append([]interface{}{}, pendingWrites...)
	}
	if checkpointHash != "" {
		out["checkpoint_hash"] = checkpointHash
	}
	if resumeMap != nil {
		out["resume_map"] = resumeMap
	}
	out["task_id"] = taskID
	return out
}

func checkpointNullVersion(_ Checkpoint) interface{} {
	// Return the zero value for int64 as the null version
	return int64(0)
}

func taskID(t interface{}) (string, bool) {
	switch v := t.(type) {
	case PregelTask:
		return v.ID, true
	case PregelExecutableTask:
		return v.ID, true
	default:
		return "", false
	}
}

func compareVersion(a, b interface{}) int {
	switch av := a.(type) {
	case int:
		bv, _ := b.(int)
		return av - bv
	case int64:
		var bv int64
		switch bvVal := b.(type) {
		case int64:
			bv = bvVal
		case int:
			bv = int64(bvVal)
		default:
			bv = 0
		}
		if av == bv {
			return 0
		}
		if av < bv {
			return -1
		}
		return 1
	case string:
		bv, _ := b.(string)
		if av == bv {
			return 0
		}
		if av < bv {
			return -1
		}
		return 1
	// Fallback to reflect.DeepEqual comparison: not perfect but safe.
	default:
		if reflect.DeepEqual(a, b) {
			return 0
		}
		return 1
	}
}
