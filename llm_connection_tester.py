import tkinter as tk
from tkinter import scrolledtext, messagebox
import json
import requests # Added for HTTP requests

# --- Default Payload ---
DEFAULT_PAYLOAD = {
    "model": "local-model",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Are you working?"}
    ],
    "temperature": 0.7
}

# --- Placeholder for Button Click Handler ---
def handle_test_connection_click():
    """
    Placeholder function for the 'Test Connection' button.
    Prints to console and updates the response_area.
    """
    # Enable response_area for updates
    response_area.config(state=tk.NORMAL)
    response_area.delete(1.0, tk.END)

    endpoint_url = url_entry.get().strip()
    headers_str = headers_entry.get().strip()
    body_str = body_text.get(1.0, tk.END).strip()

    # --- Log user inputs (for debugging the tester itself if needed) ---
    print(f"URL: {endpoint_url}")
    print(f"Headers: {headers_str}")
    print(f"Body: {body_str}")

    # --- URL Validation ---
    if not endpoint_url:
        response_area.insert(tk.END, "Error: LLM API Endpoint URL cannot be empty.\n")
        response_area.config(state=tk.DISABLED)
        return
    if not (endpoint_url.startswith("http://") or endpoint_url.startswith("https://")):
        response_area.insert(tk.END, "Error: URL must start with http:// or https://\n")
        response_area.config(state=tk.DISABLED)
        return

    # --- Parse Headers ---
    parsed_headers = {"Content-Type": "application/json"} # Default header
    if headers_str:
        try:
            user_headers = json.loads(headers_str)
            if isinstance(user_headers, dict):
                parsed_headers.update(user_headers)
            else:
                response_area.insert(tk.END, "Error: Headers must be a valid JSON object (dictionary).\n")
                response_area.config(state=tk.DISABLED)
                return
        except json.JSONDecodeError:
            response_area.insert(tk.END, "Error: Invalid JSON in Headers.\n")
            response_area.config(state=tk.DISABLED)
            return

    # --- Parse Request Body ---
    if not body_str:
        response_area.insert(tk.END, "Error: Request Body cannot be empty.\n")
        response_area.config(state=tk.DISABLED)
        return
    try:
        parsed_body = json.loads(body_str)
    except json.JSONDecodeError:
        response_area.insert(tk.END, "Error: Invalid JSON in Request Body.\n")
        response_area.config(state=tk.DISABLED)
        return

    # --- Make HTTP POST Request ---
    response_area.insert(tk.END, f"Attempting to connect to LLM at: {endpoint_url}\n...\n\n")
    response_area.see(tk.END) # Scroll to the end
    window.update_idletasks() # Force GUI update

    try:
        # Log request details to response_area
        response_area.insert(tk.END, "--- Request Details ---\n")
        response_area.insert(tk.END, f"URL: {endpoint_url}\n")
        response_area.insert(tk.END, f"Method: POST\n")
        response_area.insert(tk.END, f"Headers: {json.dumps(parsed_headers, indent=2)}\n")
        response_area.insert(tk.END, f"Body: {json.dumps(parsed_body, indent=2)}\n\n")
        response_area.see(tk.END)
        window.update_idletasks()

        response = requests.post(endpoint_url, headers=parsed_headers, json=parsed_body, timeout=20)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses

        # If successful (2xx status code)
        response_area.insert(tk.END, "--- Connection Successful! ---\n")
        response_area.insert(tk.END, f"Status Code: {response.status_code}\n\n")
        
        response_area.insert(tk.END, "--- LLM Response Headers ---\n")
        for key, value in response.headers.items():
            response_area.insert(tk.END, f"{key}: {value}\n")
        response_area.insert(tk.END, "\n")

        response_area.insert(tk.END, "--- LLM Response Body ---\n")
        try:
            llm_response_json = response.json()
            response_area.insert(tk.END, json.dumps(llm_response_json, indent=2) + "\n")
        except json.JSONDecodeError:
            response_area.insert(tk.END, "Response body is not valid JSON.\nRaw Response:\n")
            response_area.insert(tk.END, response.text + "\n")

    except requests.exceptions.HTTPError as e:
        response_area.insert(tk.END, f"--- HTTP Error ---\n")
        response_area.insert(tk.END, f"Error: {e}\n")
        if e.response is not None:
            response_area.insert(tk.END, f"Status Code: {e.response.status_code}\n")
            response_area.insert(tk.END, f"Response Body:\n{e.response.text}\n")
        else:
            response_area.insert(tk.END, "No response object available.\n")
    except requests.exceptions.ConnectionError as e:
        response_area.insert(tk.END, f"--- Connection Error ---\n")
        response_area.insert(tk.END, f"Failed to connect to the server: {e}\n")
        response_area.insert(tk.END, "Ensure the LLM server is running and the URL is correct.\n")
    except requests.exceptions.Timeout as e:
        response_area.insert(tk.END, f"--- Timeout Error ---\n")
        response_area.insert(tk.END, f"The request timed out: {e}\n")
        response_area.insert(tk.END, "Check network or if the LLM server is slow.\n")
    except requests.exceptions.RequestException as e:
        response_area.insert(tk.END, f"--- Request Error ---\n")
        response_area.insert(tk.END, f"An error occurred during the request: {e}\n")
    except json.JSONDecodeError as e: # Should be caught by the one in the success block ideally
        response_area.insert(tk.END, f"--- JSON Decode Error (LLM Response) ---\n")
        response_area.insert(tk.END, f"Failed to parse LLM's JSON response: {e}\n")
        if 'response' in locals() and response is not None:
            response_area.insert(tk.END, f"Raw Response:\n{response.text}\n")
        else:
            response_area.insert(tk.END, "No response object available to display raw text.\n")
    except Exception as e:
        response_area.insert(tk.END, f"--- An Unexpected Error Occurred ---\n")
        response_area.insert(tk.END, f"Error: {e}\n")
    finally:
        response_area.see(tk.END) # Scroll to the end
        response_area.config(state=tk.DISABLED) # Disable editing

# --- GUI Setup ---
window = tk.Tk()
window.title("LLM Connection Tester")
window.geometry("700x700") # Adjusted for better layout

# Configure grid column weights to make the entry/text widgets expand
window.columnconfigure(1, weight=1)

# --- URL Input ---
url_label = tk.Label(window, text="LLM API Endpoint URL:")
url_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
url_entry = tk.Entry(window, width=70)
url_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
url_entry.insert(0, "http://localhost:1234/v1/chat/completions") # Example URL

# --- Headers Input ---
headers_label = tk.Label(window, text="HTTP Headers (JSON, optional):")
headers_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
headers_entry = tk.Entry(window, width=70)
headers_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
headers_entry.insert(0, '{"Content-Type": "application/json"}') # Example headers

# --- Request Body Input ---
body_label = tk.Label(window, text="Request Body (JSON):")
body_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.NW) # Align label to top-west
body_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=15, width=70)
body_text.grid(row=2, column=1, padx=5, pady=5, sticky=tk.NSEW)
body_text.insert(tk.END, json.dumps(DEFAULT_PAYLOAD, indent=2))
window.rowconfigure(2, weight=1) # Allow body_text to expand vertically

# --- Test Connection Button ---
test_button = tk.Button(window, text="Test Connection", command=handle_test_connection_click)
test_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

# --- Response/Status Display Area ---
response_label = tk.Label(window, text="Response / Status:")
response_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.NW) # Align label to top-west
response_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=15, width=70, state=tk.DISABLED)
response_area.grid(row=4, column=1, padx=5, pady=5, sticky=tk.NSEW)
window.rowconfigure(4, weight=1) # Allow response_area to expand vertically

# --- Start GUI ---
if __name__ == "__main__":
    window.mainloop()
