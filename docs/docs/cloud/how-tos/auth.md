# How to add authentication to your LangGraph apps

Adding authentication to your LangGraph apps can be an important step to getting them ready for production. Once you have successfully deployed a LangGraph application, you can add authentication by sending custom headers along with every request you make to the LangGraph API - whether you are using the SDK or not.

When using the SDK you can initialize your client with the parameter `headers` and pass in the custom headers you would like. If you are sending in raw cURL commands, you can simply add in extra headers alongisde your existing ones.

Any headers that start with the prefix "x-" will automatically be passed to your graph in the configuration, and can be accessed inside any node by calling `config['configurable']['x-your-header']`. Note that headers that don't have that prefix will NOT get passed to the graph and thus will not be accessible during the graph run.

One way in which you could use custom headers for authentication is to setup a proxy that will implement your authentication logic and then forward the request to the LangGraph API if the authentication passes. In this use case you might pass a key or password to your proxy, which could then forward information such as the users name or preferences in custom headers to the LangGraph API. This keeps your authentication completely separate from your graph invocation, which can be helpful for security reasons.

[This example repository](https://github.com/langchain-ai/langgraph-with-auth) walks through setting up the proxy using Fast API, accessing custom headers inside your graph, and sends a couple example requests to show the whole pipeline in action. You can build off of this template and add your own custom authentication logic in order to safely secure access to your deployed LangGraph applications.