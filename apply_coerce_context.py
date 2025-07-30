#!/usr/bin/env python3
"""Script to apply _coerce_context in the Runtime creation locations"""

# Read the file
with open('/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py', 'r') as f:
    content = f.read()

# Replace the first occurrence (in stream method around line 2614)
old_pattern1 = """            runtime = Runtime(
                context=context,
                store=store,
                stream_writer=stream_writer,
                previous=None,
            )
            parent_runtime = config[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
            runtime = parent_runtime.merge(runtime)
            config[CONF][CONFIG_KEY_RUNTIME] = runtime

            with SyncPregelLoop("""

new_pattern1 = """            runtime = Runtime(
                context=_coerce_context(self.context_schema, context),
                store=store,
                stream_writer=stream_writer,
                previous=None,
            )
            parent_runtime = config[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
            runtime = parent_runtime.merge(runtime)
            config[CONF][CONFIG_KEY_RUNTIME] = runtime

            with SyncPregelLoop("""

# Replace the second occurrence (in astream method around line 2909)
old_pattern2 = """            runtime = Runtime(
                context=context,
                store=store,
                stream_writer=stream_writer,
                previous=None,
            )
            parent_runtime = config[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
            runtime = parent_runtime.merge(runtime)
            config[CONF][CONFIG_KEY_RUNTIME] = runtime

            async with AsyncPregelLoop("""

new_pattern2 = """            runtime = Runtime(
                context=_coerce_context(self.context_schema, context),
                store=store,
                stream_writer=stream_writer,
                previous=None,
            )
            parent_runtime = config[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
            runtime = parent_runtime.merge(runtime)
            config[CONF][CONFIG_KEY_RUNTIME] = runtime

            async with AsyncPregelLoop("""

# Apply replacements
if old_pattern1 in content:
    content = content.replace(old_pattern1, new_pattern1, 1)
    print("Updated stream method")
else:
    print("Could not find pattern for stream method")

if old_pattern2 in content:
    content = content.replace(old_pattern2, new_pattern2, 1)
    print("Updated astream method")
else:
    print("Could not find pattern for astream method")

# Write back
with open('/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py', 'w') as f:
    f.write(content)

print("Done applying _coerce_context to Runtime creations")
