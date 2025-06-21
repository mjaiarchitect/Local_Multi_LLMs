import os
import asyncio
from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, field_validator
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from graphviz import Digraph
import traceback
import sqlite3
import gradio as gr
import subprocess
import time
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt
import networkx as nx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- IMPORTANT: Ollama Model Setup Instructions ---
# This app requires three Ollama models to be installed and running:
#   1. llava:7b
#   2. qwen2.5vl:latest
#   3. nomic-embed-text
#
# To install these models, open a terminal and run:
#   ollama pull llava:7b
#   ollama pull qwen2.5vl:latest
#   ollama pull nomic-embed-text
#
# Make sure the Ollama server is running (default: http://localhost:11434).
# You can start the server with:
#   ollama serve
#
# If you have not installed Ollama, visit https://ollama.com/download to get started.

os.environ["OLLAMA_NUM_PARALLEL"] = "2"
OLLAMA_BASE_URL: str = "http://localhost:11434"
GRAPH_OUTPUT_FILE = "workflow_graph.png"

class StateSchema(BaseModel):
    message: str = Field(..., min_length=1, description="Input message or research query")
    data: Optional[str] = Field(None, description="Web search results")
    insights: Optional[str] = Field(None, description="Generated insights")
    summary: Optional[str] = Field(None, description="Final summary")
    route: Optional[str] = Field(None, description="Routing decision")

    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class ResearchWorkflow:
    def __init__(self):
        """
        This is the constructor for the ResearchWorkflow class.
        It sets up the AI models, search tool, workflow graph, database, and PDF vector database.
        This method is called automatically when you create a new ResearchWorkflow object.
        """
        self.models = {
            "llava": ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model="llava:7b"
            ),
            "maths": ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model="qwen2.5vl:latest"
            )
        }
        self.search_tool = DuckDuckGoSearchRun()
        self.graph = self._build_graph()
        self._init_db()
        self.vectordb = None

    def _init_db(self):
        """
        Initializes the SQLite database to store research results.
        This method creates a table if it doesn't exist.
        It is called automatically when the class is created.
        """
        db_path = os.path.abspath("research_results.db")
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT,
                search_results TEXT,
                insights TEXT,
                summary TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_result(self, message, search_results, insights, summary):
        """
        Saves the results of a research query to the database.
        This is called after each query is processed, so results can be viewed later.
        """
        try:
            self.cursor.execute(
                "INSERT INTO research_results (message, search_results, insights, summary) VALUES (?, ?, ?, ?)",
                (message, search_results, insights, summary)
            )
            self.conn.commit()
        except Exception as e:
            print(f"Error saving result: {e}")

    def get_last_n_results(self, n=3):
        """
        Retrieves the last n research results from the database.
        This is used to show previous results in the Gradio UI.
        """
        self.cursor.execute(
            "SELECT message, search_results, insights, summary, timestamp FROM research_results ORDER BY timestamp DESC LIMIT ?",
            (n,)
        )
        rows = self.cursor.fetchall()
        all_results = ""
        for r in rows:
            all_results += (
                f"Time: {r[4]}\n"
                f"Query: {r[0]}\n"
                f"Web Search: {r[1]}\n"
                f"Insights: {r[2]}\n"
                f"Summary: {r[3]}\n"
                f"{'-'*40}\n"
            )
        return all_results

    async def router_node(self, state: StateSchema) -> StateSchema:
        """
        Decides whether the user's query is a math problem or a general research question.
        This is the first step in the workflow graph and determines the path the query will take.
        """
        prompt = (
            "You are a router for a workflow. "
            "If the following query is a math or calculation question, respond with exactly 'Maths'. "
            "If it is anything else, respond with exactly 'Analyze'. "
            "Do not add any other words or punctuation.\n\n"
            f"Query: {state.message}"
        )
        response = await asyncio.to_thread(self.models["llava"].invoke, prompt)
        parser = StrOutputParser()
        route_decision = parser.parse(response.content).strip().lower()
        if route_decision == "maths":
            state.route = "maths"
        else:
            state.route = "analyze"
        return state

    async def maths_node(self, state: StateSchema) -> StateSchema:
        """
        Handles math or calculation queries.
        If the router_node decides the query is math-related, this node solves or analyzes the math problem.
        """
        try:
            prompt = f"Solve or analyze the following math problem:\n{state.message}"
            response = await asyncio.to_thread(self.models["maths"].invoke, prompt)
            state.insights = f"I am qwen2. {response.content.lstrip()}"
            return state
        except Exception as e:
            print(f"Error in maths node: {str(e)}")
            print(traceback.format_exc())
            raise

    async def analyze_node(self, state: StateSchema) -> StateSchema:
        """
        Handles general research queries.
        If the router_node decides the query is not math, this node performs a web search and generates insights.
        """
        try:
            search_output = await asyncio.to_thread(self.search_tool.invoke, state.message)
            state.data = search_output
            prompt = f"Based on the following web search data, generate detailed insights for the research topic:\n{search_output}"
            response = await asyncio.to_thread(self.models["llava"].invoke, prompt)
            state.insights = f"I am llava. {response.content.lstrip()}"
            return state
        except Exception as e:
            print(f"Error in analyze node: {str(e)}")
            print(traceback.format_exc())
            raise

    async def summary_node(self, state: StateSchema) -> StateSchema:
        """
        Summarizes the insights generated by either the maths_node or analyze_node.
        This is the final step in the workflow before returning results to the user.
        """
        try:
            prompt = f"Summarize the following insights for the research topic:\n{state.insights}"
            response = await asyncio.to_thread(self.models["llava"].invoke, prompt)
            state.summary = response.content
            return state
        except Exception as e:
            print(f"Error in summary generation: {str(e)}")
            print(traceback.format_exc())
            raise

    def _build_graph(self) -> StateGraph:
        """
        Builds the workflow graph that defines the steps for processing a query.
        This graph is used to decide which nodes (functions) to run for each query.
        """
        graph = StateGraph(state_schema=StateSchema)
        graph.add_node("router", self.router_node)
        graph.add_node("maths", self.maths_node)
        graph.add_node("analyze", self.analyze_node)
        graph.add_node("final_summary", self.summary_node)
        graph.add_conditional_edges(
            "router",
            lambda state: getattr(state, "route", None),
            {
                "maths": "maths",
                "analyze": "analyze",
                None: "analyze"
            }
        )
        graph.add_edge("maths", "final_summary")
        graph.add_edge("analyze", "final_summary")
        graph.add_edge("final_summary", END)
        graph.set_entry_point("router")
        return graph

    def export_graph_png(self) -> None:
        """
        Exports a PNG image of the workflow graph.
        This is used to visually show the workflow in the Gradio UI.
        """
        try:
            compiled_graph = self.graph.compile()
            nx_graph = compiled_graph.get_graph()
            dot = Digraph(comment="Research Workflow", format="png")
            for node in nx_graph.nodes:
                dot.node(str(node), str(node))
            for start, end, *_ in nx_graph.edges:
                dot.edge(str(start), str(end))
            output_file = GRAPH_OUTPUT_FILE.replace(".png", "")
            dot.render(output_file, view=False)
        except Exception as e:
            print(f"Error exporting graph: {str(e)}")
            print(traceback.format_exc())
            raise

    def export_execution_path_graph(self, route_taken: str, filename: str = "execution_path_graph.png") -> None:
        """
        Exports a PNG image showing the path taken through the workflow for a specific query.
        The path is highlighted in green. This helps users see how their query was processed.
        """
        try:
            compiled_graph = self.graph.compile()
            nx_graph = compiled_graph.get_graph()
            dot = Digraph(comment="Execution Path", format="png")
            for node in nx_graph.nodes:
                dot.node(str(node), str(node))
            if route_taken == "maths":
                green_edges = [("router", "maths"), ("maths", "final_summary"), ("final_summary", "END")]
            else:
                green_edges = [("router", "analyze"), ("analyze", "final_summary"), ("final_summary", "END")]
            for start, end, *_ in nx_graph.edges:
                if (str(start), str(end)) in green_edges:
                    dot.edge(str(start), str(end), color="green", penwidth="3")
                else:
                    dot.edge(str(start), str(end), color="black")
            output_file = filename.replace(".png", "")
            dot.render(output_file, view=False)
        except Exception as e:
            print(f"Error exporting execution path graph: {str(e)}")
            print(traceback.format_exc())
            raise

    async def analyze(self, message: str) -> Dict[str, Any]:
        """
        Main function to process a user's research or math query.
        It runs the workflow graph and returns the results.
        This is called when the user submits a query in the Gradio UI.
        """
        try:
            state = StateSchema(message=message)
            app = self.graph.compile()
            result = await app.ainvoke(state)
            return result
        except Exception as e:
            print(f"Error in research workflow: {str(e)}")
            print(traceback.format_exc())
            raise

    def load_pdf_and_create_vectorstore(self, pdf_path: str):
        """
        Loads a PDF file, splits it into chunks, and creates a vector database for RAG (Retrieval-Augmented Generation).
        This allows the app to answer questions based on the content of uploaded PDFs.
        Called when a user uploads PDFs in the Gradio UI.
        """
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
        # If vectordb exists, add new docs; else create new
        if self.vectordb is None:
            vectordb = Chroma.from_documents(splits, embeddings, persist_directory="rag_chroma_db")
            vectordb.persist()
            self.vectordb = vectordb
        else:
            self.vectordb.add_documents(splits)
            self.vectordb.persist()

    def rag_query(self, query: str) -> str:
        """
        Answers a user's question using only the content from the uploaded PDFs.
        This is called when the user asks a question in the PDF RAG tab.
        """
        if not hasattr(self, "vectordb") or self.vectordb is None:
            return "No PDF loaded for RAG."
        retriever = self.vectordb.as_retriever()
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = (
            f"Answer the following question using ONLY the provided context from the PDF.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        response = self.models["llava"].invoke(prompt)
        return response.content

# --- Gradio Functions ---

def analyze_and_display(query):
    """
    This function is called when the user submits a research or math query in the Gradio UI.
    It runs the workflow, saves the result, updates the execution path graph, and returns all outputs for display.
    """
    workflow = ResearchWorkflow()
    result = asyncio.run(workflow.analyze(query.strip()))
    route_taken = result.get("route", "analyze")
    exec_graph_path = "execution_path_graph.png"  # Always overwrite the same file
    workflow.export_execution_path_graph(route_taken, filename=exec_graph_path)
    workflow.save_result(
        query,
        result.get("data"),
        result.get("insights"),
        result.get("summary")
    )
    all_results = workflow.get_last_n_results(3)
    return (
        result.get("data"),
        result.get("insights"),
        result.get("summary"),
        all_results,
        "workflow_graph.png",
        exec_graph_path
    )

# --- PDF RAG page functions ---
rag_workflow = ResearchWorkflow()

def handle_pdfs(pdf_files):
    """
    Handles PDF uploads from the user in the Gradio UI.
    Loads and indexes each PDF for RAG-based question answering.
    """
    for pdf in pdf_files:
        rag_workflow.load_pdf_and_create_vectorstore(pdf.name)
    return f"{len(pdf_files)} PDF(s) loaded and indexed for RAG!"

def handle_rag_query(rag_query):
    """
    Handles questions about the uploaded PDFs.
    Uses the indexed PDFs to answer the user's question.
    """
    return rag_workflow.rag_query(rag_query)

# --- Gradio UI ---

rag_workflow.export_graph_png()  # Ensure workflow_graph.png exists before UI

with gr.Blocks() as demo:
    with gr.Tab("PDF RAG"):
        pdf_files = gr.File(label="Upload PDF(s) for RAG", file_count="multiple", type="filepath")
        pdf_status = gr.Textbox(label="PDF Status")
        rag_query_box = gr.Textbox(label="Ask a question about the PDF(s)")
        rag_enter = gr.Button("Enter")
        rag_answer = gr.Textbox(label="RAG Answer", lines=5)
        pdf_files.upload(handle_pdfs, inputs=pdf_files, outputs=pdf_status)
        rag_enter.click(handle_rag_query, inputs=rag_query_box, outputs=rag_answer)
        rag_query_box.submit(handle_rag_query, inputs=rag_query_box, outputs=rag_answer)

    with gr.Tab("Research/Maths Assistant"):
        with gr.Row():
            query = gr.Textbox(label="Enter your research or maths query")
            submit = gr.Button("Submit")
        with gr.Row():
            search_results = gr.Textbox(label="Web Search Results", lines=10, max_lines=30, show_copy_button=True)
            insights = gr.Textbox(label="Insights", lines=5)
            summary = gr.Textbox(label="Summary", lines=5)
        all_results = gr.Textbox(
            label="Previous 3 Results",
            lines=10, max_lines=30, show_copy_button=True
        )
        with gr.Row():
            workflow_graph = gr.Image(label="Workflow Graph")
            execution_path_graph = gr.Image(label="Execution Path Graph")

        submit.click(
            analyze_and_display,
            inputs=query,
            outputs=[search_results, insights, summary, all_results, workflow_graph, execution_path_graph]
        )
        

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
