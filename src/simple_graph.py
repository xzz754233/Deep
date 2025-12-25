import asyncio
import json  # <--- 新增：為了格式化輸出 JSON
from typing import TypedDict, List, Annotated
import operator

from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# Import existing modules to ensure compatibility
from src.state import Chronology, ChronologyEvent
from src.url_crawler.utils import scrape_page_content
from src.llm_service import create_llm_structured_model
from src.utils import get_langfuse_handler

# --- 1. State Definition ---
class SimpleState(TypedDict):
    """
    State definition for the lightweight (Lite) mode.
    """
    person_to_research: str
    urls: List[str]
    raw_content: str
    structured_events: List[ChronologyEvent]

# --- 2. Node Logic ---

def simple_search_node(state: SimpleState):
    """
    Step 1: Search using Tavily, limiting to top 2 results for speed.
    """
    person = state["person_to_research"]
    print(f"\n🚀 [Simple Mode] Starting search: {person}")
    
    # Append keywords to improve precision
    query = f"{person} biography timeline life events"
    
    # max_results=2 to save time and tokens
    tool = TavilySearch(max_results=2)
    results = tool.invoke({"query": query})
    
    # Extract URLs
    urls = [r["url"] for r in results["results"]]
    print(f"🔍 Found {len(urls)} URLs: {urls}")
    
    return {"urls": urls}


async def simple_scrape_node(state: SimpleState):
    """
    Step 2: Scrape all URLs in parallel and merge into one large text block.
    """
    urls = state.get("urls", [])
    print(f"🕷️ [Simple Mode] Starting scrape for {len(urls)} pages...")
    
    # Define an internal function to handle single URL errors safely
    async def safe_scrape(url):
        try:
            print(f"   Scraping: {url} ...")
            content = await scrape_page_content(url)
            if content:
                # Add source markers for LLM context
                return f"=== Source: {url} ===\n{content}\n"
        except Exception as e:
            print(f"⚠️ Scrape failed for {url}: {e}")
        return ""

    # Use asyncio to process all scrape tasks in parallel
    tasks = [safe_scrape(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Filter empty strings and merge
    full_text = "\n".join([r for r in results if r])
    print(f"📄 Scraping complete, total characters: {len(full_text)}")
    
    return {"raw_content": full_text}


async def simple_extract_node(state: SimpleState, config: RunnableConfig):
    """
    Step 3: One-shot extraction of structured JSON using LLM.
    """
    print("🧠 [Simple Mode] Starting event extraction (LLM)...")
    content = state.get("raw_content", "")
    
    if not content:
        print("❌ No content to extract.")
        return {"structured_events": []}

    # Define Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an expert biographical researcher. Read the following raw data about {person} and extract key life events.

    <Rules>
    1. The output MUST strictly follow the JSON structure (containing name, description, date, location).
    2. Order events chronologically.
    3. Include birth, education, key career milestones, personal life (marriage/children), major achievements, and death/legacy.
    4. Keep descriptions concise but informative.
    5. If a field (like location) is not mentioned in the text, leave it blank/null.
    </Rules>

    <Raw Data>
    {text}
    </Raw Data>
    """)

    # Bind the project's existing structured model configuration
    # class_name=Chronology ensures output matches the original JSON format
    model = create_llm_structured_model(config, class_name=Chronology)
    
    chain = prompt | model
    
    # Execute LLM
    try:
        result = await chain.ainvoke({
            "person": state["person_to_research"],
            # Truncate to avoid context limit (though usually fine for Gemini/GPT-4o)
            "text": content[:100000] 
        })
        
        events = result.events
        print(f"✅ Extraction successful, found {len(events)} events.")

        # --- 👇 新增：直接在終端機印出結果 👇 ---
        print("\n" + "="*40)
        print(f"🎉 Final Result for {state['person_to_research']}:")
        # 將 Pydantic 物件轉為 Dict 再轉 JSON 字串，以確保格式美觀
        serializable_events = [e.model_dump() for e in events]
        
        print("\n" + "="*40)
        print(f"🎉 Final Result for {state['person_to_research']}:")
        print(json.dumps(serializable_events, indent=2, ensure_ascii=False))
        print("="*40 + "\n")

        # ↓ 正確：回傳轉換好的 Dict (JSON)
        return {"structured_events": serializable_events}
        
    except Exception as e:
        print(f"❌ LLM Extraction failed: {e}")
        return {"structured_events": []}


# --- 3. Graph Assembly ---

workflow = StateGraph(SimpleState)

# Add Nodes
workflow.add_node("search", simple_search_node)
workflow.add_node("scrape", simple_scrape_node)
workflow.add_node("extract", simple_extract_node)

# Create Linear Flow
workflow.add_edge(START, "search")
workflow.add_edge("search", "scrape")
workflow.add_edge("scrape", "extract")
workflow.add_edge("extract", END)

# Compile Graph
simple_graph = workflow.compile().with_config({"callbacks": [get_langfuse_handler()]})