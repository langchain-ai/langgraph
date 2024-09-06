# How to kick off background runs

<div class="admonition tip">
    <p class="admonition-title">Setup <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how LangSmith can help you <a href="https://docs.smith.langchain.com
        ">here</a>. 
    </p>
</div>    

This guide covers how to kick off background runs for your agent.
This can be useful for long running jobs.

## Setup

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

Output:

    {
        'thread_id': '5cb1e8a1-34b3-4a61-a34e-71a9799bd00d',
        'created_at': '2024-08-30T20:35:52.062934+00:00',
        'updated_at': '2024-08-30T20:35:52.062934+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

## Check runs on thread

If we list the current runs on this thread, we will see that it's empty:

=== "Python"

    ```python
    runs = await client.runs.list(thread["thread_id"])
    print(runs)
    ```

=== "Javascript"

    ```js
    let runs = await client.runs.list(thread['thread_id']);
    console.log(runs);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs
    ```

Output:

    []

## Start runs on thread

Now let's kick off a run:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    run = await client.runs.create(thread["thread_id"], assistant_id, input=input)
    ```

=== "Javascript"

    ```js
    let input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]};
    let run = await client.runs.create(thread["thread_id"], assistantID, { input });
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>
        }'
    ```

The first time we poll it, we can see `status=pending`:

=== "Python"

    ```python
    print(await client.runs.get(thread["thread_id"], run["run_id"]))
    ```

=== "Javascript"

    ```js
    console.log(await client.runs.get(thread["thread_id"], run["run_id"]));
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>
    ```

Output:

        {
            "run_id": "1ef6a5f8-bd86-6763-bbd6-bff042db7b1b",
            "thread_id": "7885f0cf-94ad-4040-91d7-73f7ba007c8a",
            "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
            "created_at": "2024-09-04T01:46:47.244887+00:00",
            "updated_at": "2024-09-04T01:46:47.244887+00:00",
            "metadata": {},
            "status": "pending",
            "kwargs": {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "what's the weather in sf"
                        }
                    ]
                },
                "config": {
                    "metadata": {
                        "created_by": "system"
                    },
                    "configurable": {
                        "run_id": "1ef6a5f8-bd86-6763-bbd6-bff042db7b1b",
                        "user_id": "",
                        "graph_id": "agent",
                        "thread_id": "7885f0cf-94ad-4040-91d7-73f7ba007c8a",
                        "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
                        "checkpoint_id": null
                    }
                },
                "webhook": null,
                "temporary": false,
                "stream_mode": [
                    "values"
                ],
                "feedback_keys": null,
                "interrupt_after": null,
                "interrupt_before": null
            },
            "multitask_strategy": "reject"
        }



Now we can join the run, wait for it to finish and check that status again:

=== "Python"

    ```python
    await client.runs.join(thread["thread_id"], run["run_id"])
    print(await client.runs.get(thread["thread_id"], run["run_id"]))
    ```

=== "Javascript"

    ```js
    await client.runs.join(thread["thread_id"], run["run_id"]);
    console.log(await client.runs.get(thread["thread_id"], run["run_id"]));
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/join &&
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>
    ```

Output:

    {
        "run_id": "1ef6a5f8-bd86-6763-bbd6-bff042db7b1b",
        "thread_id": "7885f0cf-94ad-4040-91d7-73f7ba007c8a",
        "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
        "created_at": "2024-09-04T01:46:47.244887+00:00",
        "updated_at": "2024-09-04T01:46:47.244887+00:00",
        "metadata": {},
        "status": "success",
        "kwargs": {
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "what's the weather in sf"
                    }
                ]
            },
            "config": {
                "metadata": {
                    "created_by": "system"
                },
                "configurable": {
                    "run_id": "1ef6a5f8-bd86-6763-bbd6-bff042db7b1b",
                    "user_id": "",
                    "graph_id": "agent",
                    "thread_id": "7885f0cf-94ad-4040-91d7-73f7ba007c8a",
                    "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
                    "checkpoint_id": null
                }
            },
            "webhook": null,
            "temporary": false,
            "stream_mode": [
                "values"
            ],
            "feedback_keys": null,
            "interrupt_after": null,
            "interrupt_before": null
        },
        "multitask_strategy": "reject"
    }


Perfect! The run succeeded as we would expect. We can double check that the run worked as expected by printing out the final state:

=== "Python"

    ```python
    final_result = await client.threads.get_state(thread["thread_id"])
    print(final_result)
    ```

=== "Javascript"

    ```js
    let finalResult = await client.threads.getState(thread["thread_id"]);
    console.log(finalResult);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state
    ```

Output:

    {
        "values": {
            "messages": [
                {
                    "content": "what's the weather in sf",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": null,
                    "id": "beba31bf-320d-4125-9c37-cadf526ac47a",
                    "example": false
                },
                {
                    "content": [
                        {
                            "id": "toolu_01AaNPSPzqia21v7aAKwbKYm",
                            "input": {},
                            "name": "tavily_search_results_json",
                            "type": "tool_use",
                            "index": 0,
                            "partial_json": "{\"query\": \"weather in san francisco\"}"
                        }
                    ],
                    "additional_kwargs": {},
                    "response_metadata": {
                        "stop_reason": "tool_use",
                        "stop_sequence": null
                    },
                    "type": "ai",
                    "name": null,
                    "id": "run-f220faf8-1d27-4f73-ad91-6bb3f47e8639",
                    "example": false,
                    "tool_calls": [
                        {
                            "name": "tavily_search_results_json",
                            "args": {
                                "query": "weather in san francisco"
                            },
                            "id": "toolu_01AaNPSPzqia21v7aAKwbKYm",
                            "type": "tool_call"
                        }
                    ],
                    "invalid_tool_calls": [],
                    "usage_metadata": {
                        "input_tokens": 273,
                        "output_tokens": 61,
                        "total_tokens": 334
                    }
                },
                {
                    "content": "[{\"url\": \"https://www.weatherapi.com/\", \"content\": \"{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1725052131, 'localtime': '2024-08-30 14:08'}, 'current': {'last_updated_epoch': 1725051600, 'last_updated': '2024-08-30 14:00', 'temp_c': 21.1, 'temp_f': 70.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 11.9, 'wind_kph': 19.1, 'wind_degree': 290, 'wind_dir': 'WNW', 'pressure_mb': 1018.0, 'pressure_in': 30.07, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 59, 'cloud': 25, 'feelslike_c': 21.1, 'feelslike_f': 70.0, 'windchill_c': 18.6, 'windchill_f': 65.5, 'heatindex_c': 18.6, 'heatindex_f': 65.5, 'dewpoint_c': 12.2, 'dewpoint_f': 54.0, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 15.0, 'gust_kph': 24.2}}\"}]",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "tool",
                    "name": "tavily_search_results_json",
                    "id": "686b2487-f332-4e58-9508-89b3a814cd81",
                    "tool_call_id": "toolu_01AaNPSPzqia21v7aAKwbKYm",
                    "artifact": {
                        "query": "weather in san francisco",
                        "follow_up_questions": null,
                        "answer": null,
                        "images": [],
                        "results": [
                            {
                                "title": "Weather in San Francisco",
                                "url": "https://www.weatherapi.com/",
                                "content": "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1725052131, 'localtime': '2024-08-30 14:08'}, 'current': {'last_updated_epoch': 1725051600, 'last_updated': '2024-08-30 14:00', 'temp_c': 21.1, 'temp_f': 70.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 11.9, 'wind_kph': 19.1, 'wind_degree': 290, 'wind_dir': 'WNW', 'pressure_mb': 1018.0, 'pressure_in': 30.07, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 59, 'cloud': 25, 'feelslike_c': 21.1, 'feelslike_f': 70.0, 'windchill_c': 18.6, 'windchill_f': 65.5, 'heatindex_c': 18.6, 'heatindex_f': 65.5, 'dewpoint_c': 12.2, 'dewpoint_f': 54.0, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 15.0, 'gust_kph': 24.2}}",
                                "score": 0.976148,
                                "raw_content": null
                            }
                        ],
                        "response_time": 3.07
                    },
                    "status": "success"
                },
                {
                    "content": [
                        {
                            "text": "\n\nThe search results provide the current weather conditions in San Francisco. According to the data, as of 2:00 PM on August 30, 2024, the temperature in San Francisco is 70\u00b0F (21.1\u00b0C) with partly cloudy skies. The wind is blowing from the west-northwest at around 12 mph (19 km/h). The humidity is 59% and visibility is 9 miles (16 km). Overall, it looks like a nice late summer day in San Francisco with comfortable temperatures and partly sunny conditions.",
                            "type": "text",
                            "index": 0
                        }
                    ],
                    "additional_kwargs": {},
                    "response_metadata": {
                        "stop_reason": "end_turn",
                        "stop_sequence": null
                    },
                    "type": "ai",
                    "name": null,
                    "id": "run-8fecc61d-3d9f-4e16-8e8a-92f702be498a",
                    "example": false,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "usage_metadata": {
                        "input_tokens": 837,
                        "output_tokens": 124,
                        "total_tokens": 961
                    }
                }
            ]
        },
        "next": [],
        "tasks": [],
        "metadata": {
            "step": 3,
            "run_id": "1ef67140-eb23-684b-8253-91d4c90bb05e",
            "source": "loop",
            "writes": {
                "agent": {
                    "messages": [
                        {
                            "id": "run-8fecc61d-3d9f-4e16-8e8a-92f702be498a",
                            "name": null,
                            "type": "ai",
                            "content": [
                                {
                                    "text": "\n\nThe search results provide the current weather conditions in San Francisco. According to the data, as of 2:00 PM on August 30, 2024, the temperature in San Francisco is 70\u00b0F (21.1\u00b0C) with partly cloudy skies. The wind is blowing from the west-northwest at around 12 mph (19 km/h). The humidity is 59% and visibility is 9 miles (16 km). Overall, it looks like a nice late summer day in San Francisco with comfortable temperatures and partly sunny conditions.",
                                    "type": "text",
                                    "index": 0
                                }
                            ],
                            "example": false,
                            "tool_calls": [],
                            "usage_metadata": {
                                "input_tokens": 837,
                                "total_tokens": 961,
                                "output_tokens": 124
                            },
                            "additional_kwargs": {},
                            "response_metadata": {
                                "stop_reason": "end_turn",
                                "stop_sequence": null
                            },
                            "invalid_tool_calls": []
                        }
                    ]
                }
            },
            "user_id": "",
            "graph_id": "agent",
            "thread_id": "5cb1e8a1-34b3-4a61-a34e-71a9799bd00d",
            "created_by": "system",
            "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca"
        },
        "created_at": "2024-08-30T21:09:00.079909+00:00",
        "checkpoint_id": "1ef67141-3ca2-6fae-8003-fe96832e57d6",
        "parent_checkpoint_id": "1ef67141-2129-6b37-8002-61fc3bf69cb5"
    }

We can also just print the content of the last AIMessage:

=== "Python"

    ```python
    print(final_result['values']['messages'][-1]['content'][0]['text'])
    ```

=== "Javascript"

    ```js
    console.log(finalResult['values']['messages'][finalResult['values']['messages'].length-1]['content'][0]['text']);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | jq -r '.values.messages[-1].content.[0].text'
    ```

Output:

    The search results provide the current weather conditions in San Francisco. According to the data, as of 2:00 PM on August 30, 2024, the temperature in San Francisco is 70°F (21.1°C) with partly cloudy skies. The wind is blowing from the west-northwest at around 12 mph (19 km/h). The humidity is 59% and visibility is 9 miles (16 km). Overall, it looks like a nice late summer day in San Francisco with comfortable temperatures and partly sunny conditions.