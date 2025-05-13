## Cron jobs

There are many situations in which it is useful to run an assistant on a schedule. 

For example, say that you're building an assistant that runs daily and sends an email summary
of the day's news. You could use a cron job to run the assistant every day at 8:00 PM.

LangGraph Platform supports cron jobs, which run on a user-defined schedule. The user specifies a schedule, an assistant, and some input. After that, on the specified schedule, the server will:

- Create a new thread with the specified assistant
- Send the specified input to that thread

Note that this sends the same input to the thread every time. See the [how-to guide](../../cloud/how-tos/cron_jobs.md) for creating cron jobs.

The LangGraph Platform API provides several endpoints for creating and managing cron jobs. See the [API reference](../../cloud/reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/crons) for more details.