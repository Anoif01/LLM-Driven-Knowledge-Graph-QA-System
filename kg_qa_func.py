# pipeline.py
import pandas as pd
import numpy as np
import re
import json
import difflib
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# ------------------- Data loading -------------------
def load_triples(file_path='./data/triples.csv'):
    triple_df = pd.read_csv(file_path)
    triples = list(zip(triple_df["Subject"], triple_df["Predicate"], triple_df["Object"]))
    return triples

# ------------------- Model loading -------------------

def load_qa_pipeline(model_name="google/gemma-2b-it"):
    """
    Load a text generation model and tokenizer for QA task.
    
    Args:
        model_name (str): The model name or path on Huggingface.
    
    Returns:
        pipeline: Huggingface text-generation pipeline
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return qa_pipeline

# ------------------- Funcs -------------------

def build_prompt(question):
    return f"""
  Extract the entity, predicate, entity_type based on the question. 
  Only return one JSON with your reason. Do not generat the question and no-given contents.

  Given the question below, extract:
      - entity: (the key entity mentioned, could be a movie title, a person, a genre, or a year)
      
      - predicate: (e.g. directed_by, starred_by, has_genre, released_on, released_year)
      - entity_type: (the type of the extracted entity, could be person, movie, a genre, a year )

  Example Q1: Who directed Inception?
  → {{"entity": "Inception", "entity_type":"movie", predicate": "directed_by"}}

  Example Q2: Which films did Tom Hanks act in?
  → {{"entity": "Tom Hanks", "entity_type":"person", "predicate": "starred_by"}}

  Example Q3: Munin Barua is director of which films?
  → {{"entity": "Munin Barua", "entity_type":"person", "predicate": "directed_by"}}

  Example Q4: Show me the movies released in 2022.
  → {{"entity": "2022", "entity_type":"year", "predicate": "released_year"}}

  Example Q4: Show me the genre of Cyher.
  → {{"entity": "Cypher", "entity_type":"movie", "predicate": "has_genre"}}


  Now answer for:
  Q: {question}
→
"""

def parse_query_with_gemma(question, qa_pipeline):
    input_prompt = build_prompt(question)
    output = qa_pipeline(
        input_prompt,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.3,
        repetition_penalty=1.2
    )[0]["generated_text"]

    json_matches = re.findall(r"\{.*?\}", output, re.DOTALL)
    if not json_matches:
        return {"error": "No JSON found"}

    last_json = json_matches[-1]
    try:
        return json.loads(last_json)
    except json.JSONDecodeError:
        return {"error": "JSON decode failed", "raw": last_json}

def fuzzy_match(entity, target, threshold=0.8):
    seq = difflib.SequenceMatcher(None, entity.lower(), target.lower())
    return seq.ratio() >= threshold

def structured_kg_search(parsed, triples):
    results = []
    related_graphs = []

    predicate = parsed.get("predicate", "")
    entity_type = parsed.get("entity_type", "").lower()
    entity = parsed.get("entity", "").lower()

    # Inferring direction based on entity type
    if entity_type == 'movie':
        direction = 'forward'
    else:
        direction = 'backward'

    for s, p, o in triples:
        if not fuzzy_match(predicate, p):
            continue

        if direction == "forward" and fuzzy_match(entity, s):
            results.append((s, p, o))
        elif direction == "backward" and fuzzy_match(entity, o):
            results.append((s, p, o))

    for s, p, o in triples:
        if direction == "forward" and fuzzy_match(entity, s):
            related_graphs.append((s, p, o))
        elif direction == "backward" and fuzzy_match(entity, o):
            related_graphs.append((s, p, o))

    return results, direction, related_graphs

def generate_answer(question, file_path='./data/triples.csv', model_name="google/gemma-2b-it"):
    triples = load_triples(file_path)
    qa_pipeline = load_qa_pipeline(model_name)

    parsed = parse_query_with_gemma(question, qa_pipeline)
    print("LLM Output:", parsed, '\n')

    if "error" in parsed:
        return "Sorry, I couldn't parse your question."

    facts, direction, related_graphs = structured_kg_search(parsed, triples)
    parsed['direction'] = direction
    # print("Matched Triples:", facts, '\n')

    if not facts:
        return parsed, facts, f"{parsed['entity']} is not in the Knowledge Graphs. Try another question.", related_graphs
        #, "Sorry, I couldn't find an answer."
    elif direction == "forward":
        objs = [o for s, p, o in facts]
        objs = ", ".join(objs)
        return parsed, facts, f"{facts[0][0]} → {facts[0][1]} → {objs}", related_graphs
    else:
        movies = [s for s, p, o in facts]
        if parsed['entity_type'] == 'person':
            pred = parsed['predicate'].replace('_', ' ')
            return parsed, facts, f"Movies {pred} {parsed['entity']}: {', '.join(movies)}", related_graphs
        elif parsed['entity_type'] == 'genre':
            return parsed, facts, f"{parsed['entity']} movies: {', '.join(movies)}", related_graphs
        elif parsed['entity_type'] == 'year':
            return parsed, facts, f"Movies in {parsed['entity']}: {', '.join(movies)}", related_graphs
        else:
            return parsed, facts, ", ".join(movies), related_graphs

# ------------------- Test -------------------

if __name__ == "__main__":
    test_question = "Which films did Tom Hanks act in?"
    answer = generate_answer(test_question, triples)
    print("Answer:\n", answer)
