import sys
import json
import time
import requests
from datetime import datetime
from langgraph.graph import StateGraph, END
from ollama_client import query_ollama
from typing import TypedDict
from retrying import retry
from tqdm import tqdm
from duckduckgo_search import DDGS
from flask import Flask, render_template, request, jsonify, Response
import threading
import queue

app = Flask(__name__)

class AgentState(TypedDict):
    query: str
    model: str
    research: str
    investigation: str
    analysis: str
    strategy: str
    action_plan: str
    full_output: str  # Stores concatenated output for TTS

# Global variable to store the latest state
latest_state = None

# --- Retry decorator for Ollama API ---
def retry_if_connection_error(exception):
    return isinstance(exception, (requests.ConnectionError, requests.Timeout))

@retry(retry_on_exception=retry_if_connection_error, stop_max_attempt_number=3, wait_fixed=2000)
def query_ollama_stream(prompt, model="mistral", output_queue=None):
    """
    Streams output from Ollama line by line with retry logic.
    Adapted for streaming to a queue for web UI.
    """
    url = "http://127.0.0.1:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "max_tokens": 300}
    response = requests.post(url, json=payload, stream=True, timeout=10)
    
    output = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                if "response" in data:
                    chunk = data["response"]
                    output += chunk
                    if output_queue:
                        output_queue.put(chunk)
            except json.JSONDecodeError:
                continue
    if output_queue:
        output_queue.put(None)  # Signal end of stream
    return output

# --- Web Research Helper ---
def perform_web_research(query: str, num_results: int = 5) -> str:
    """
    Performs a web search using DuckDuckGo and returns summarized snippets.
    """
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
        
        snippets = []
        for result in results:
            # Only include results with non-empty title and body
            if result['title'] and result['body']:
                snippets.append(f"Title: {result['title']}\nSnippet: {result['body']}\nURL: {result['href']}\n")
        
        research_summary = "\n".join(snippets)
        return research_summary if snippets else ""
    except Exception as e:
        return ""

# --- Agents (Adapted for queue-based streaming) ---
def researcher_agent(state: AgentState, output_queue):
    query = state.get("query")
    model = state.get("model", "mistral")
    if not query:
        raise ValueError("No query provided!")
    
    output_queue.put("=== Web Research ===\n")
    state["full_output"] = state.get("full_output", "") + "=== Web Research ===\n"
    research = perform_web_research(query)
    
    if research:
        # Check if research results are meaningful
        if len(research.strip()) > 50:  # Arbitrary threshold for meaningful content
            output_queue.put("Summarizing web research...\n")
            state["full_output"] += "Summarizing web research...\n"
            summary_prompt = f"Summarize the key insights from this web research relevant to the query '{query}':\n{research}"
            research_summary = query_ollama_stream(summary_prompt, model, output_queue)
            state["research"] = research_summary
            state["full_output"] += research_summary + "\n"
        else:
            output_queue.put("No suitable web research results found. Proceeding with model-based response.\n")
            state["full_output"] += "No suitable web research results found. Proceeding with model-based response.\n"
            state["research"] = ""
    else:
        output_queue.put("No suitable web research results found. Proceeding with model-based response.\n")
        state["full_output"] += "No suitable web research results found. Proceeding with model-based response.\n"
        state["research"] = ""
    
    return state

def investigator_agent(state: AgentState, output_queue):
    query = state.get("query")
    research = state.get("research", "")
    model = state.get("model", "mistral")
    if not query:
        raise ValueError("No query provided!")
    
    output_queue.put("=== Investigation ===\n")
    state["full_output"] = state.get("full_output", "") + "=== Investigation ===\n"
    research_context = f"\nWeb Research Summary: {research}" if research else ""
    info = query_ollama_stream(f"Investigate this problem carefully and report all key facts, incorporating any relevant web research{research_context}:\n{query}", model, output_queue)
    state["investigation"] = info
    state["full_output"] += info + "\n"
    return state

def analyst_agent(state: AgentState, output_queue):
    investigation = state.get("investigation")
    model = state.get("model", "mistral")
    if not investigation:
        raise ValueError("No investigation found!")
    
    output_queue.put("=== Analysis ===\n")
    state["full_output"] = state.get("full_output", "") + "=== Analysis ===\n"
    analysis = query_ollama_stream(f"Analyze this investigation and find hidden patterns or overlooked issues:\n{investigation}", model, output_queue)
    state["analysis"] = analysis
    state["full_output"] += analysis + "\n"
    return state

def strategist_agent(state: AgentState, output_queue):
    analysis = state.get("analysis")
    model = state.get("model", "mistral")
    if not analysis:
        raise ValueError("No analysis found!")
    
    output_queue.put("=== Strategy ===\n")
    state["full_output"] = state.get("full_output", "") + "=== Strategy ===\n"
    strategy = query_ollama_stream(f"Based on this analysis, suggest the most practical and creative solutions:\n{analysis}", model, output_queue)
    state["strategy"] = strategy
    state["full_output"] += strategy + "\n"
    return state

def advisor_agent(state: AgentState, output_queue):
    strategy = state.get("strategy")
    model = state.get("model", "mistral")
    if not strategy:
        raise ValueError("No strategy found!")
    
    output_queue.put("=== Action Plan ===\n")
    state["full_output"] = state.get("full_output", "") + "=== Action Plan ===\n"
    plan = query_ollama_stream(f"Turn this strategy into a clear step-by-step action plan:\n{strategy}", model, output_queue)
    state["action_plan"] = plan
    state["full_output"] += plan + "\n"
    return state

# --- Build Graph ---
graph = StateGraph(AgentState)
graph.add_node("researcher", lambda state: researcher_agent(state, output_queue))
graph.add_node("investigator", lambda state: investigator_agent(state, output_queue))
graph.add_node("analyst", lambda state: analyst_agent(state, output_queue))
graph.add_node("strategist", lambda state: strategist_agent(state, output_queue))
graph.add_node("advisor", lambda state: advisor_agent(state, output_queue))

graph.add_edge("researcher", "investigator")
graph.add_edge("investigator", "analyst")
graph.add_edge("analyst", "strategist")
graph.add_edge("strategist", "advisor")
graph.add_edge("advisor", END)

graph.set_entry_point("researcher")
app_graph = graph.compile()

# Global queue for streaming output
output_queue = queue.Queue()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global latest_state
    user_query = request.form.get('query')
    model_choice = request.form.get('model', 'mistral')
    
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Clear queue
    while not output_queue.empty():
        output_queue.get()
    
    initial_state = AgentState({"query": user_query, "model": model_choice, "full_output": ""})
    
    # Run the graph in a thread
    def run_workflow():
        global latest_state
        try:
            result = app_graph.invoke(initial_state)
            latest_state = result  # Store the final state
            output_queue.put("=== Workflow Complete ===\n")
            output_queue.put(None)  # End signal
        except Exception as e:
            output_queue.put(f"Error: {str(e)}\n")
            output_queue.put(None)
    
    threading.Thread(target=run_workflow).start()
    
    return jsonify({'message': 'Processing started'})

@app.route('/stream')
def stream():
    def generate():
        while True:
            chunk = output_queue.get()
            if chunk is None:
                break
            yield f"data: {chunk}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/generate-audio', methods=['GET'])
def generate_audio():
    global latest_state
    if not latest_state or not latest_state.get("full_output"):
        return jsonify({'error': 'No output available for audio generation'}), 400
    return jsonify({'text': latest_state["full_output"]})

if __name__ == "__main__":
    app.run(debug=True)