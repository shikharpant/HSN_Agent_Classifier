# --- Imports ---
import os
import time
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display, Markdown
import json
from tqdm.notebook import tqdm

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List

# --- Load Environment Variables ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    display(Markdown("### ⚠️ **Action Required**\nGoogle API key not found. Please create a `.env` file with your `GOOGLE_API_KEY` and restart the kernel."))
else:
    display(Markdown("### ✅ API Key Loaded Successfully"))



# --- LLM Initialization ---
# Using default timeout handling which is more stable with recent library versions.
# Robust chunking and error handling are our primary strategies.

classifier_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    temperature=0.2, # Slightly higher temp to encourage exploring options
    model_kwargs={"response_mime_type": "application/json"} # Enforce JSON output
)

# Gemini 2.5 Pro: More powerful and precise for the final, authoritative review
reviewer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0, # Zero temperature for a deterministic final decision
    model_kwargs={"response_mime_type": "application/json"}
)
display(Markdown("### ✅ LLMs initialized: Classifier (Flash) and Reviewer (Pro)"))

# --- Pydantic Models for Structured Output ---
class HsnCandidate(BaseModel):
    hsn_code: str = Field(description="The most likely 6 or 8-digit HSN code.")
    confidence: int = Field(description="Classifier's confidence score (0-100).")
    reasoning: str = Field(description="Brief reasoning for the choice.")

class ClassifierBatchOutput(BaseModel):
    classifications: List[HsnCandidate]

class ReviewerItemOutput(BaseModel):
    s_no: int = Field(description="The original serial number of the product.")
    final_hsn_code: str = Field(description="The single, definitive HSN code.")
    final_reasoning: str = Field(description="Consolidated, final reasoning for the choice.")
    final_confidence: int = Field(description="Reviewer's final confidence score.")

class ReviewerBatchOutput(BaseModel):
    reviewed_items: List[ReviewerItemOutput]

display(Markdown("### ✅ Pydantic models defined"))

# --- Output Parsers ---
classifier_parser = JsonOutputParser(pydantic_object=ClassifierBatchOutput)
reviewer_batch_parser = JsonOutputParser(pydantic_object=ReviewerBatchOutput)

# --- Agent 1: Batch Classifier (Gemini Flash) - Prompt ---
classifier_prompt = ChatPromptTemplate.from_template(
    """
    **Role:** You are an efficient HSN Classification bot processing a batch of products.
    **Task:** For each product description provided in the input JSON, determine the single most likely HSN code.

    **CRITICAL INSTRUCTIONS:**
    1.  You MUST return a classification for every single item in the input batch.
    2.  The order of your output list MUST EXACTLY match the order of the input list.
    3.  Keep your reasoning CONCISE and to the point.
    4.  After processing the last item in the batch, you MUST stop and produce no further text.

    **Input Batch of Products (JSON format):**
    ```json
    {product_batch_json}
    ```

    **Your output MUST be a JSON object that strictly follows this format and nothing else:**
    {format_instructions}
    """
)

# --- Agent 2: Batch Reviewer (Gemini Pro) - Prompt (Added for completeness) ---
reviewer_batch_prompt = ChatPromptTemplate.from_template(
    """
    **Role:** You are a meticulous HSN Code Auditor.
    **Task:** Review a batch of product classifications that have already been assigned a preliminary HSN code. Your job is to act as a final quality check.

    **Instructions:**
    1.  Carefully examine each product's description and the suggested HSN code.
    2.  If you agree with the suggestion, confirm the HSN code and provide a confidence score of 95 or higher.
    3.  If you disagree or believe a more accurate HSN code exists, provide the corrected HSN code and explain your reasoning.
    4.  Your reasoning should be brief but clear, justifying your final decision.
    5.  You MUST return a review for every single item in the input batch. The order of your output must match the input order.

    **Input Batch for Review (JSON format):**
    ```json
    {review_batch_json}
    ```

    **Your output MUST be a JSON object that strictly follows this format and nothing else:**
    {format_instructions}
    """
)

# --- Main Processing Function (Corrected) ---
def create_chunks(df, chunk_size=20):
    """Splits the dataframe into smaller, manageable chunks of a fixed size."""
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

def process_hsn_classification_from_excel(filepath: str, confidence_threshold: int = 80):
    try:
        df_input = pd.read_excel(filepath)
    except FileNotFoundError:
        display(Markdown(f"### ❌ Error: The file `{filepath}` was not found."))
        return None, None
    display(Markdown(f"✅ Loaded `{filepath}` with **{len(df_input)}** rows."))

    # Standardize column names
    df_input.rename(columns={'S. No.': 'S.No.', 'item_description': 'Description'}, inplace=True, errors='ignore')

    reviewed_results, manual_review_list = [], []
    description_map = df_input.set_index('S.No.')['Description'].to_dict()
    classifier_map = {}

    chunks = create_chunks(df_input, chunk_size=20)
    display(Markdown(f"Divided the input into **{len(chunks)}** chunk(s) of up to 10 items each."))

    for i, chunk_df in enumerate(tqdm(chunks, desc="Processing Chunks")):
        display(Markdown(f"---\n### 🕵️ **Processing Chunk {i+1}/{len(chunks)}** ({len(chunk_df)} rows) with Classifier..."))

        batch_input_list = chunk_df[['S.No.', 'Description']].to_dict(orient='records')
        product_batch_json = json.dumps([{"s_no": r['S.No.'], "description": r['Description']} for r in batch_input_list])

        classifier_output = None
        try:
            raw_classifier_response = classifier_llm.invoke(classifier_prompt.format(
                product_batch_json=product_batch_json,
                format_instructions=classifier_parser.get_format_instructions()
            ))
            
            # --- FIX: The parser returns a dict. We must instantiate our Pydantic object from it. ---
            parsed_dict = classifier_parser.parse(raw_classifier_response.content)
            classifier_output = ClassifierBatchOutput(**parsed_dict)


        except Exception as e:
            display(Markdown(f"**❌ Error classifying chunk {i+1}:** {e}."))
            display(Markdown("Adding all items from this chunk to manual review."))
            for _, row in chunk_df.iterrows():
                manual_review_list.append({'S.No.': row['S.No.'], 'Description': row['Description'], 'HSN (classifier)': 'CLASSIFICATION_ERROR', 'Confidence (classifier)': 0})
            continue

        if not classifier_output or not classifier_output.classifications:
            display(Markdown(f"**❌ Error: Classifier returned a null or empty response for chunk {i+1}.** Adding all items to manual review."))
            for _, row in chunk_df.iterrows():
                manual_review_list.append({'S.No.': row['S.No.'], 'Description': row['Description'], 'HSN (classifier)': 'CLASSIFICATION_NULL_RESPONSE', 'Confidence (classifier)': 0})
            continue

        items_for_review = []
        processed_s_nos = set()

        # Now, this dot notation access works correctly because classifier_output is a Pydantic object.
        returned_classifications = classifier_output.classifications

        if len(returned_classifications) != len(chunk_df):
            display(Markdown(f"**⚠️ Warning: Classifier output size mismatch in chunk {i+1}.** Expected {len(chunk_df)}, got {len(returned_classifications)}. Salvaging what we can."))

        for index, classified_item in enumerate(returned_classifications):
            if index >= len(chunk_df):
                continue

            original_row = chunk_df.iloc[index]
            s_no = int(original_row['S.No.'])
            description = original_row['Description']
            processed_s_nos.add(s_no)

            # This dot notation now works correctly because classified_item is an HsnCandidate object.
            classifier_data = {
                's_no': s_no,
                'hsn_code': classified_item.hsn_code,
                'confidence': classified_item.confidence,
                'reasoning': classified_item.reasoning
            }
            classifier_map[s_no] = classifier_data

            if classifier_data['confidence'] > confidence_threshold:
                items_for_review.append({
                    "s_no": s_no, "description": description,
                    "suggested_hsn": classifier_data['hsn_code'],
                    "classifier_reasoning": classifier_data['reasoning']
                })
            else:
                manual_review_list.append({
                    'S.No.': s_no, 'Description': description,
                    'HSN (classifier)': classifier_data['hsn_code'],
                    'Confidence (classifier)': classifier_data['confidence']
                })

        all_chunk_s_nos = set(chunk_df['S.No.'])
        missed_s_nos = all_chunk_s_nos - processed_s_nos

        if missed_s_nos:
            display(Markdown(f"**[INFO] {len(missed_s_nos)} items were missed by the classifier in this chunk. Sending to manual review.**"))
            for s_no in missed_s_nos:
                manual_review_list.append({
                    'S.No.': s_no, 'Description': description_map.get(s_no, "N/A"),
                    'HSN (classifier)': 'MISSED_BY_LLM', 'Confidence (classifier)': 0
                })

        if items_for_review:
            display(Markdown(f"▷ **Reviewing {len(items_for_review)} high-confidence items** with Pro model..."))
            review_batch_json = json.dumps(items_for_review)

            try:
                raw_reviewer_response = reviewer_llm.invoke(reviewer_batch_prompt.format(
                    review_batch_json=review_batch_json,
                    format_instructions=reviewer_batch_parser.get_format_instructions()
                ))

                # --- FIX: Same as above. Instantiate the ReviewerBatchOutput from the parsed dict. ---
                parsed_reviewer_dict = reviewer_batch_parser.parse(raw_reviewer_response.content)
                reviewer_batch_output = ReviewerBatchOutput(**parsed_reviewer_dict)

                # Now, this loop works as intended.
                for reviewed_item in reviewer_batch_output.reviewed_items:
                    s_no = reviewed_item.s_no
                    classifier_data = classifier_map.get(s_no, {})
                    reviewed_results.append({
                        'S.No.': s_no, 'Description': description_map.get(s_no, "N/A"),
                        'HSN_Final': reviewed_item.final_hsn_code,
                        'Confidence_Final': reviewed_item.final_confidence,
                        'Classifier_HSN': classifier_data.get('hsn_code', 'N/A'),
                        'Classifier_Confidence': classifier_data.get('confidence', 'N/A'),
                    })
                display(Markdown(f"✅ Batch review complete. Pausing briefly..."))
                time.sleep(2)
            except Exception as e:
                display(Markdown(f"**❌ Error during batch review for Chunk {i+1}:** {e}. Adding all items to manual review."))
                for item in items_for_review:
                    s_no = item['s_no']
                    classifier_data = classifier_map.get(s_no, {})
                    manual_review_list.append({'S.No.': s_no, 'Description': description_map.get(s_no, "N/A"), 'HSN (classifier)': classifier_data.get('hsn_code', 'N/A'), 'Confidence (classifier)': classifier_data.get('confidence', 'N/A')})

    display(Markdown("--- \n ## ✅ Processing Complete. Generating output files..."))

    df_reviewed = pd.DataFrame(reviewed_results)
    if not df_reviewed.empty:
        df_reviewed.to_csv('reviewed_classifications.csv', index=False)
        display(Markdown(f"### Reviewed Classifications (`{len(df_reviewed)}` items)"))
        display(df_reviewed)

    df_manual_review = pd.DataFrame(manual_review_list)
    if not df_manual_review.empty:
        df_manual_review.sort_values(by='S.No.', inplace=True)
        df_manual_review.to_csv('manual_review_needed.csv', index=False)
        display(Markdown(f"### Manual Review Needed (`{len(df_manual_review)}` items)"))
        display(df_manual_review)

    return df_reviewed, df_manual_review


# --- Run the pipeline ---
excel_filepath = 'benchmark1000.xlsx'
# Ensure you have a file with 'S.No.' and 'Description' columns
if os.path.exists(excel_filepath):
    try:
        pd.read_excel(excel_filepath, sheet_name=0)[['S. No.', 'Description']]
    except (KeyError, FileNotFoundError):
        print(f"no such file exists")
        
    df_reviewed, df_manual_review = process_hsn_classification_from_excel(excel_filepath)
else:
    display(Markdown(f"### ❌ Error: Input file `{excel_filepath}` not found. Please create it and run this cell again."))
