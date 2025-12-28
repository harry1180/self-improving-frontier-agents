# File: sifa_prototype.py

# Requirements: pip install openai langchain sentence-transformers scikit-learn hdbscan pandas

import os

import json

import numpy as np

import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.cluster import HDBSCAN

from openai import OpenAI

from langchain.prompts import PromptTemplate

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Replace with frontier model endpoint

# 1. SIFA Inference Loop (Section 5.1 pseudocode)

def sifa_inference(user_query, retrieved_context, tools_available=True):

    system_prompt = """

    You are a governed insurance claims adjudication agent. 

    Always reason step-by-step, cite policy clauses, and enforce regulatory compliance.

    """

    

    messages = [

        {"role": "system", "content": system_prompt},

        {"role": "user", "content": user_query},

        {"role": "assistant", "content": f"Retrieved context: {retrieved_context}"}

    ]

    

    # Simulate tool call capability

    response = client.chat.completions.create(

        model="gpt-4o",  # Replace with o3/claude opus endpoint

        messages=messages,

        tools=[] if not tools_available else [{"type": "function", "function": {"name": "get_policy", "description": "Retrieve policy"}}],

        temperature=0.0

    )

    

    return response.choices[0].message.content

# 2. Feedback Capture Schema (Section 5.2)

feedback_schema = {

    "claim_id": "str",

    "reasoning_trace_id": "str",

    "issue_type": "str",

    "human_correction": "str",

    "severity": "high|medium|low",

    "timestamp": "str"

}

# Example feedback entries

feedback_data = [

    {"claim_id": "78421", "issue_type": "policy_misinterpretation", "human_correction": "Correct clause is 3.2(c) not 4.1(b)", "severity": "high"},

    {"claim_id": "78422", "issue_type": "policy_misinterpretation", "human_correction": "Clause 3.2(c) applies for water damage", "severity": "high"},

    {"claim_id": "78423", "issue_type": "eligibility_error", "human_correction": "Deductible already met", "severity": "medium"}

]

# 3. Pattern Mining (Section 5.3)

def mine_patterns(feedback_df):

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(feedback_df['human_correction'].tolist())

    

    clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean')

    labels = clusterer.fit_predict(embeddings)

    feedback_df['cluster'] = labels

    

    # Identify high-impact clusters

    patterns = feedback_df[feedback_df['cluster'] != -1].groupby('cluster').agg({

        'issue_type': 'first',

        'human_correction': list,

        'severity': 'mean'

    }).sort_values('severity', ascending=False)

    

    return patterns

# 4. Synthetic Data Generation (Section 5.4)

synthetic_prompt = PromptTemplate.from_template("""

Correct the following reasoning error and generate 3 diverse counterfactual variants targeting the same failure mode:

Error: {error}

Original correction: {correction}

Output format: JSON list of synthetic examples.

""")

def generate_synthetic(example):

    response = client.chat.completions.create(

        model="gpt-4o",

        messages=[{"role": "user", "content": synthetic_prompt.format(error=example['issue_type'], correction=example['human_correction'])}],

        temperature=0.7

    )

    return json.loads(response.choices[0].message.content)

# Example usage

if __name__ == "__main__":

    query = "Is water damage from burst pipe covered under policy POL-2025-001?"

    context = "Policy clause 3.2(c): Sudden and accidental discharge from plumbing covered..."

    print("Inference:", sifa_inference(query, context))

    

    df = pd.DataFrame(feedback_data)

    patterns = mine_patterns(df)

    print("\nMined Patterns:\n", patterns)

    

    if not patterns.empty:

        top_error = patterns.iloc[0]

        synthetics = generate_synthetic(top_error)

        print("\nSynthetic Examples:", synthetics)
