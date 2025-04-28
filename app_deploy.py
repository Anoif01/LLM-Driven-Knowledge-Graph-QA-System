import streamlit as st
from kg_qa_func import load_qa_pipeline, generate_answer, load_triples

import networkx as nx
import matplotlib.pyplot as plt

def plot_kg_triples(triples, all_triples):
    if not triples:
        st.warning("No triples to visualize.")
        return

    G = nx.DiGraph()

    # Add all triples to the graph
    for s, p, o in all_triples:
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, label=p)

    # --- Highlight matched nodes ---
    matched_nodes = set()
    for s, p, o in triples:
        matched_nodes.add(s)
        matched_nodes.add(o)

    node_colors = []
    for node in G.nodes():
        if node in matched_nodes:
            node_colors.append("salmon")  # Highlight matched nodes (orange-red)
        else:
            node_colors.append("lightblue")  # Other nodes (light blue)

    # Plot the graph
    pos = nx.spring_layout(G, k=0.6, seed=42)  # Layout style: spring layout

    plt.figure(figsize=(5, 3))
    nx.draw(G, pos, with_labels=True, node_size=50, node_color=node_colors, font_size=5, font_weight="bold", arrowsize=5, edge_color="lightgray")

    # Overlay matched triples (highlighted in red)
    if triples:
        highlight_edges = [(s, o) for s, p, o in triples]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=highlight_edges,
            edge_color="salmon",
            width=2,
            arrows=True,
            arrowsize=5
        )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

    plt.margins(x=0.3, y=0.15)  # Add margins to avoid cutting labels
    st.pyplot(plt.gcf())
    plt.clf()  # Clear figure to avoid overlapping drawings

# --- Page title and description ---
st.set_page_config(page_title="LLM-Driven Knowledge Graph QA System", layout="wide")
st.title("🎬 LLM-Driven Knowledge Graph QA System")

st.markdown(
    """
    ### 📌 About the System:
    This system uses a Large Language Model to understand user questions and translate them into Knowledge Graph queries.
    It directly retrieves facts from a curated movie knowledge graph, providing answers that are **accurate, explainable, and trustworthy**.
    The Movie Knowledge Graph is constructed from **DBpedia**[1], focusing on films released between **2000 and 2023**.

    ### ⚙️ Key Components:
    - **LLM Intent Parsing**: User queries are converted into Knowledge Graph query languages using the LLM **Gemma 2B-It**.
    - **Knowledge Graph Retrieval**: Fact-based retrieval from the Movie Knowledge Graph.
    - **Answer Generation**: Generated answers grounded in structured facts.
    """
)

# --- Horizontal separator ---
st.markdown("---")

# --- Create two columns layout ---
col1, spacer, col2 = st.columns([5, 1, 5])

# --- Left column (User input + Final answer) ---
with col1:
    st.header("🔎 Ask a Question")
    user_question = st.text_input("Ask a question about movies:")

    if st.button("📤 Search", key="search_button"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🔍 Analyzing your question with LLM..."):
                parsed_info, triples_matched, answer, related_graphs = generate_answer(user_question)

            # --- LLM intent parsing result ---
            st.markdown("---")
            if 'parsed_info' in locals():
                st.header("🧠 LLM Intent Parsing")

                st.markdown(f"""
                              <div style="background-color: #F0F2F6; padding: 10px; border-radius: 15px;">
                              <strong>Question:</strong><br>
                              <code>{user_question}</code><br><br>
                              <strong>Transformed to:</strong><br>
                              <code>{parsed_info['entity']} → {parsed_info['predicate']} → ?</code>
                              </div>
                              """,
                              unsafe_allow_html=True
                          )
            else:
                st.warning(
                f"⚠️ No parsing result. The LLM could not convert this query.\n\n"
            )

            # --- Final answer ---
            st.markdown("---")
            st.header("✅ Final Answer")
            if triples_matched:
                st.success(answer)
            else:
                st.warning(
                f"⚠️ No matching information found. The Knowledge Graph does not contain sufficient details for this query.\n\n"
            )

# --- Right column (Knowledge graph retrieval and visualization) ---
with col2:
    if 'triples_matched' in locals():
        st.header("📚 Knowledge Graph Retrieval")
        if triples_matched:
            # Visualize knowledge graph triples
            plot_kg_triples(triples_matched, related_graphs)
        else:
            st.warning("No relevant triples found in the Knowledge Graph.")

# --- Reference section at the bottom ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.8em; color: gray;'>
    Reference: [1] Lehmann, J., et al. (2015).
    <i>DBpedia – A Large-scale, Multilingual Knowledge Base Extracted from Wikipedia.</i>
    Semantic Web Journal, 6(2), 167–195.
    <a href='https://svn.aksw.org/papers/2013/SWJ_DBpedia/public.pdf' target='_blank'>Link</a>
    </div>
    """,
    unsafe_allow_html=True
)
