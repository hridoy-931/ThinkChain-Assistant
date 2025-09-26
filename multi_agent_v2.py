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

class AgentState(TypedDict):
    query: str
    model: str
    research: str
    investigation: str
    analysis: str
    strategy: str
    action_plan: str

# --- Retry decorator for Ollama API ---
def retry_if_connection_error(exception):
    return isinstance(exception, (requests.ConnectionError, requests.Timeout))

@retry(retry_on_exception=retry_if_connection_error, stop_max_attempt_number=3, wait_fixed=2000)
def query_ollama_stream(prompt, model="mistral"):
    """
    Streams output from Ollama line by line with retry logic.
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
                    print(chunk, end="", flush=True)
            except json.JSONDecodeError:
                continue
    print()  # newline at the end
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
            snippets.append(f"Title: {result['title']}\nSnippet: {result['body']}\nURL: {result['href']}\n")
        
        research_summary = "\n".join(snippets)
        print(f"\nFound {len(results)} web results for '{query}'.")
        return research_summary
    except Exception as e:
        print(f"Web search failed: {str(e)}. Falling back to no research.")
        return ""

# --- Save output to JSON ---
def save_output_to_json(state: AgentState):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "timestamp": timestamp,
        "query": state.get("query", ""),
        "model": state.get("model", ""),
        "research": state.get("research", ""),
        "investigation": state.get("investigation", ""),
        "analysis": state.get("analysis", ""),
        "strategy": state.get("strategy", ""),
        "action_plan": state.get("action_plan", "")
    }
    filename = f"output_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nOutput saved to {filename}")

# --- Agents ---
def researcher_agent(state: AgentState):
    query = state.get("query")
    model = state.get("model", "mistral")
    if not query:
        raise ValueError("No query provided!")
    
    print("\n=== Web Research ===")
    with tqdm(total=100, desc="Researching", bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
        research = perform_web_research(query)
        pbar.update(100)
    
    if research:
        # Summarize research using Ollama
        print("\nSummarizing web research...")
        summary_prompt = f"Summarize the key insights from this web research relevant to the query '{query}':\n{research}"
        research_summary = query_ollama_stream(summary_prompt, model)
        state["research"] = research_summary
    else:
        state["research"] = ""
    
    return state

def investigator_agent(state: AgentState):
    query = state.get("query")
    research = state.get("research", "")
    model = state.get("model", "mistral")
    if not query:
        raise ValueError("No query provided!")
    
    print("\n=== Investigation ===")
    research_context = f"\nWeb Research Summary: {research}" if research else ""
    with tqdm(total=100, desc="Investigating", bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
        info = query_ollama_stream(f"Investigate this problem carefully and report all key facts, incorporating any relevant web research{research_context}:\n{query}", model)
        pbar.update(100)
    state["investigation"] = info
    return state

def analyst_agent(state: AgentState):
    investigation = state.get("investigation")
    model = state.get("model", "mistral")
    if not investigation:
        raise ValueError("No investigation found!")
    
    print("\n=== Analysis ===")
    with tqdm(total=100, desc="Analyzing", bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
        analysis = query_ollama_stream(f"Analyze this investigation and find hidden patterns or overlooked issues:\n{investigation}", model)
        pbar.update(100)
    state["analysis"] = analysis
    return state

def strategist_agent(state: AgentState):
    analysis = state.get("analysis")
    model = state.get("model", "mistral")
    if not analysis:
        raise ValueError("No analysis found!")
    
    print("\n=== Strategy ===")
    with tqdm(total=100, desc="Strategizing", bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
        strategy = query_ollama_stream(f"Based on this analysis, suggest the most practical and creative solutions:\n{analysis}", model)
        pbar.update(100)
    state["strategy"] = strategy
    return state

def advisor_agent(state: AgentState):
    strategy = state.get("strategy")
    model = state.get("model", "mistral")
    if not strategy:
        raise ValueError("No strategy found!")
    
    print("\n=== Action Plan ===")
    with tqdm(total=100, desc="Planning", bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
        plan = query_ollama_stream(f"Turn this strategy into a clear step-by-step action plan:\n{strategy}", model)
        pbar.update(100)
    state["action_plan"] = plan
    return state

# --- Build Graph ---
graph = StateGraph(AgentState)
graph.add_node("researcher", researcher_agent)
graph.add_node("investigator", investigator_agent)
graph.add_node("analyst", analyst_agent)
graph.add_node("strategist", strategist_agent)
graph.add_node("advisor", advisor_agent)

graph.add_edge("researcher", "investigator")
graph.add_edge("investigator", "analyst")
graph.add_edge("analyst", "strategist")
graph.add_edge("strategist", "advisor")
graph.add_edge("advisor", END)

graph.set_entry_point("researcher")
app = graph.compile()

# --- Run Workflow ---
if __name__ == "__main__":
    # Get user input
    user_query = input("Enter your problem: ").strip()
    if not user_query:
        print("Error: Please provide a valid problem.")
        sys.exit(1)
    
    # Get model selection
    model_choice = input("Enter the Ollama model to use (e.g., mistral, llama3, or press Enter for default 'mistral'): ").strip() or "mistral"
    available_models = ["mistral", "llama3"]  # Add more models as needed
    if model_choice not in available_models:
        print(f"Warning: Model '{model_choice}' not in known models {available_models}. Using default 'mistral'.")
        model_choice = "mistral"

    initial_state = AgentState({"query": user_query, "model": model_choice})
    print(f"\nStarting workflow with query: '{user_query}' and model: '{model_choice}'")
    
    try:
        result = app.invoke(initial_state)
        save_output_to_json(result)
        print("\nFinal Result:", result)
    except Exception as e:
        print(f"\nError during workflow execution: {str(e)}")
        sys.exit(1)