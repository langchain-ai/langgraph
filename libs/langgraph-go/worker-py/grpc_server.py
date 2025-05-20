import concurrent.futures
import logging
import sys
import time
from typing import Dict, Callable, Iterator, Dict

from stubs import server_pb2, server_pb2_grpc
import grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class WorkerServicer(server_pb2_grpc.WorkerServicer):
    """Implementation of the Worker service."""
    
    def __init__(self):
        # You might want to initialize resources here
        self.node_handlers: Dict[str, Callable] = {}
        self.reducer_handlers: Dict[str, Callable] = {}
    
    def register_node_handler(self, name: str, handler: Callable):
        """Register a handler for a specific node."""
        self.node_handlers[name] = handler
    
    def register_reducer_handler(self, name: str, handler: Callable):
        """Register a handler for a specific reducer."""
        self.reducer_handlers[name] = handler
    
    def StreamNode(self, request: server_pb2.PregelExecutableTask, 
                  context: grpc.ServicerContext) -> Iterator[server_pb2.Event]:
        """Call stream on a task.
        
        Args:
            request: The PregelExecutableTask containing task details
            context: The gRPC context
            
        Yields:
            Event messages with write or error events
        """
        logger.info(f"StreamNode called with task_id: {request.task_id}, name: {request.name}")
        
        try:
            # Check if we have a handler for this node
            if request.name not in self.node_handlers:
                error_msg = f"No handler registered for node: {request.name}"
                logger.error(error_msg)
                # Return an error event
                yield self._create_error_event("handler_not_found", error_msg.encode())
                return
            
            # Call the handler
            handler = self.node_handlers[request.name]
            
            # Process inputs (you may need to deserialize them based on your needs)
            inputs = request.input
            
            # Call the handler and process its results
            results = handler(inputs, request.config, request.path)
            
            # Yield results as Event messages
            for name, value in results.items():
                yield self._create_write_event(name, value)
                
        except Exception as e:
            logger.exception(f"Error in StreamNode: {str(e)}")
            yield self._create_error_event("internal_error", str(e).encode())
    
    def InvokeReducer(self, request: server_pb2.PregelExecutableTask, 
                     context: grpc.ServicerContext) -> Iterator[server_pb2.Event]:
        """Invoke a reducer.
        
        Args:
            request: The PregelExecutableTask containing task details
            context: The gRPC context
            
        Yields:
            Event messages with write or error events
        """
        logger.info(f"InvokeReducer called with task_id: {request.task_id}, name: {request.name}")
        
        try:
            # Check if we have a handler for this reducer
            if request.name not in self.reducer_handlers:
                error_msg = f"No handler registered for reducer: {request.name}"
                logger.error(error_msg)
                # Return an error event
                yield self._create_error_event("handler_not_found", error_msg.encode())
                return
            
            # Call the handler
            handler = self.reducer_handlers[request.name]
            
            # Process inputs (you may need to deserialize them based on your needs)
            inputs = request.input
            
            # Call the handler and process its results
            results = handler(inputs, request.config, request.path)
            
            # Yield results as Event messages
            for name, value in results.items():
                yield self._create_write_event(name, value)
                
        except Exception as e:
            logger.exception(f"Error in InvokeReducer: {str(e)}")
            yield self._create_error_event("internal_error", str(e).encode())
    
    def _create_write_event(self, name: str, value: bytes) -> server_pb2.Event:
        """Create a write event."""
        event = server_pb2.Event()
        event.write.name = name
        event.write.value = value
        return event
    
    def _create_error_event(self, name: str, value: bytes) -> server_pb2.Event:
        """Create an error event."""
        event = server_pb2.Event()
        event.error.name = name
        event.error.value = value
        return event


def serve(port: int = 50051, max_workers: int = 10):
    """Start the gRPC server.
    
    Args:
        port: The port to listen on
        max_workers: Maximum number of worker threads
    """
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    )
    
    # Create and register the servicer
    servicer = WorkerServicer()
    server_pb2_grpc.add_WorkerServicer_to_server(servicer, server)
    
    # Add a secure port (you might want to add proper credentials in production)
    server.add_insecure_port(f'[::]:{port}')
    
    # Start the server
    server.start()
    logger.info(f"Server started, listening on port {port}")
    
    # Keep the server running until interrupted
    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


def register_handlers(servicer: WorkerServicer):
    """Register handlers for nodes and reducers.
    
    This is where you would register your custom handlers for different
    node types and reducers.
    
    Args:
        servicer: The WorkerServicer instance
    """
    # Example node handler
    def example_node_handler(inputs, config, path):
        # Process inputs and return results
        # This is just a placeholder implementation
        return {"result": b"Example node result"}
    
    # Example reducer handler
    def example_reducer_handler(inputs, config, path):
        # Process inputs and return results
        # This is just a placeholder implementation
        return {"result": b"Example reducer result"}
    
    # Register handlers
    servicer.register_node_handler("example_node", example_node_handler)
    servicer.register_reducer_handler("example_reducer", example_reducer_handler)


def main():
    """Main entry point."""
    # Parse command line arguments if needed
    port = 50051
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    # Create the servicer
    servicer = WorkerServicer()
    
    # Register handlers
    register_handlers(servicer)
    
    # Start the server
    serve(port=port)


if __name__ == "__main__":
    main()
