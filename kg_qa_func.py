# pipeline.py
import pandas as pd
import numpy as np
import re
import json
import difflib
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import ast

# ------------------- Data Loading -------------------
def load_triples(file_path='./20002017_triples.csv'):
    triple_df = pd.read_csv(file_path)
    triples = list(zip(triple_df["Subject"], triple_df["Predicate"], triple_df["Object"]))
    return triples
    # triple_df = pd.read_csv(file_path)
    # triple_df['Subject'] = triple_df['Subject'].apply(lambda x: re.sub(r" \(.*?\)", '', x))
    # triple_df['Object'] = triple_df['Object'].apply(lambda x: re.sub(r" \(.*?\)", '', x))
    # triples = list(zip(triple_df["Subject"], triple_df["Predicate"], triple_df["Object"]))
    # return triples

# ------------------- Model Loading -------------------

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

# ------------------- Functions -------------------

def build_prompt(question):
    return f"""
  Extract the entity, predicate, entity_type based on the question. 
  Only return one JSON with your reason. Do not generat the question and no-given contents.

  Given the question below, extract:
      - entity: (the key entity mentioned, could be a movie title, a person, a genre, or a year)
      
      - predicate: (e.g. directed_by, starred_by, has_genre, released_on_date, released_on_year)
      - entity_type: (the type of the extracted entity, could be person, movie, a genre, a date, a year )

  Example Q1: When did Inception release?
  → {{"entity": "Inception", "entity_type":"movie", predicate": "released_on_date"}}

  Example Q2: Which films did Tom Hanks act in?
  → {{"entity": "Tom Hanks", "entity_type":"person", "predicate": "starred_by"}}

  Example Q3: Munin Barua is director of which films?
  → {{"entity": "Munin Barua", "entity_type":"person", "predicate": "directed_by"}}

  Example Q4: Show me the movies released in 2022.
  → {{"entity": "2022", "entity_type":"year", "predicate": "released_on_year"}}

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

def strip_brackets(text, bracket_pattern):
    """Remove the brackets and their contents"""
    return bracket_pattern.sub('', text).strip()

def remove_bracket_symbols_only(text):
    """Remove only the bracket symbols and retain the contents"""
    return re.sub(r'[()]', '', text).strip()

def split_bracket_parts(text):
    """Extract the contents outside the brackets and inside the brackets,"""
    match = re.search(r"(.*?)\s*\((.*?)\)(.*)", text)
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    else:
        return text.strip(), None

def fuzzy_match(entity, target, threshold=0.9):
    # regex for Parentheses
    bracket_pattern = re.compile(r"\(.*?\)")

    s = entity.strip().lower()
    m = target.strip().lower()

    # Case 1: If there are no parentheses, match directly.
    if '(' not in m:
        # score = fuzz.token_sort_ratio(s, m)
        seq = difflib.SequenceMatcher(None, s, m)
        return seq.ratio() >= threshold

    # Case 2: When parentheses exist
    ## Remove all bracket contents
    m_no_brackets = strip_brackets(m, bracket_pattern)
    score_no_brackets = difflib.SequenceMatcher(None, s, m_no_brackets)
    if score_no_brackets.ratio() >= threshold:
        return True

    ## Remove only the bracket
    m_keep_content = remove_bracket_symbols_only(m)
    score_keep_content = difflib.SequenceMatcher(None, s, m_keep_content)
    if score_keep_content.ratio() >= threshold:
        return True

    ## Extract the contents outside and inside the brackets
    m1, m2, m3 = split_bracket_parts(m)
    if m1 != '':
        score_m1 = difflib.SequenceMatcher(None, s, m1).ratio()
    else:
        score_m1 = 0
    
    if m2 != '':
        score_m2 = difflib.SequenceMatcher(None, s, m2).ratio()
    else:
        score_m2 = 0
    
    if m3 != '':
        score_m3 = difflib.SequenceMatcher(None, s, m3).ratio()
    else:
        score_m3 = 0

    if max(score_m1, score_m2, score_m2) >= threshold:
        return True

    # ALL fail
    return False

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

        # if no matched triples, maybe direction is wrong 
        if results == [] and direction == "forward":
            if fuzzy_match(entity, o):
                results.append((s, p, o))
                direction = "backward"
        elif results == [] and direction == "backward":
            if fuzzy_match(entity, s):
                results.append((s, p, o))
                direction = "forward"

    for s, p, o in triples:
        if direction == "forward" and fuzzy_match(entity, s):
            related_graphs.append((s, p, o))
        elif direction == "backward" and fuzzy_match(entity, o):
            related_graphs.append((s, p, o))

    return results, direction, related_graphs

def generate_answer(question, file_path='./20002017_triples.csv', model_name="google/gemma-2b-it"):
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
