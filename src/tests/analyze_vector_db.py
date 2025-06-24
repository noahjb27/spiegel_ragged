import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging
import tiktoken  # Import the tokenizer library

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Project-Specific Imports ---
try:
    from src.core.vector_store import ChromaDBInterface
except ImportError as e:
    print(f"Error: Failed to import ChromaDBInterface.")
    print(f"Please ensure this script is in the root of your project directory and that all dependencies are installed.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
OUTPUT_DIR = "analysis_results"

# NEW: Define the specific collections to be analyzed as requested
TARGET_COLLECTIONS = [
    'recursive_chunks_3000_300_TH_cosine_nomic-embed-text',
    'recursive_chunks_2000_400_TH_cosine_nomic-embed-text',
    'recursive_chunks_500_100_TH_cosine_nomic-embed-text'
]

# --- Main Analysis Function ---
def analyze_chromadb():
    """
    Connects to ChromaDB and analyzes only the specified collections,
    calculating chunk counts, unique articles, and total tokens.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Results will be saved in '{OUTPUT_DIR}'.")
    logging.info(f"Targeting specific collections: {TARGET_COLLECTIONS}")

    try:
        # Initialize the tokenizer for counting tokens
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logging.error(f"Could not initialize tiktoken tokenizer: {e}")
        logging.error("Please install it using 'pip install tiktoken'")
        return

    try:
        # 1. Connect to ChromaDB using the project's interface
        logging.info("Initializing ChromaDBInterface to connect to the database...")
        db_interface = ChromaDBInterface()
        client = db_interface.client
        client.heartbeat()
        logging.info("Successfully connected to ChromaDB.")

        all_collections = client.list_collections()
        all_collection_names = [c.name for c in all_collections]
        
        # Filter to get only the collections we care about
        collections_to_process = [c for c in all_collections if c.name in TARGET_COLLECTIONS]

        if not collections_to_process:
            logging.warning("None of the target collections were found in the database.")
            logging.info(f"Available collections are: {all_collection_names}")
            return

        summary_data = []

        # 2. Analyze each targeted collection
        for collection in collections_to_process:
            collection_name = collection.name
            logging.info(f"\n--- Analyzing Collection: {collection_name} ---")

            num_chunks = collection.count()
            if num_chunks == 0:
                logging.warning(f"Collection '{collection_name}' is empty. Skipping.")
                continue

            logging.info(f"Fetching documents and metadata for {num_chunks} chunks...")
            # Include 'documents' to get the text content for token counting
            data = collection.get(include=["documents", "metadatas"])
            
            # --- Calculations ---
            # Total Chunks
            num_chunks = len(data["ids"])

            # Unique Articles
            df = pd.DataFrame(data["metadatas"])
            num_unique_articles = df['Artikeltitel'].nunique() if 'Artikeltitel' in df.columns else 0
            
            # Total Tokens
            logging.info("Calculating total tokens...")
            total_tokens = sum(len(tokenizer.encode(doc)) for doc in data["documents"])
            logging.info(f"Token calculation complete.")

            # Yearly Distribution
            if 'Jahrgang' in df.columns:
                df['Jahrgang'] = pd.to_numeric(df['Jahrgang'], errors='coerce')
                yearly_counts = df.dropna(subset=['Jahrgang'])['Jahrgang'].astype(int).value_counts().sort_index()
            else:
                yearly_counts = pd.Series(dtype=int)

            summary_data.append({
                "Collection Name": collection_name,
                "Total Chunks": num_chunks,
                "Total Tokens": total_tokens,
                "Unique Articles": num_unique_articles,
            })

            logging.info(f"Total chunks: {num_chunks}")
            logging.info(f"Total tokens: {total_tokens:,}") # Formatted with comma
            logging.info(f"Unique articles: {num_unique_articles}")

            # 3. Generate and save visualization (optional, can be commented out)
            if not yearly_counts.empty:
                plt.figure(figsize=(12, 7))
                yearly_counts.plot(kind='bar', color='teal')
                plt.title(f'Article Distribution per Year in: {collection_name}', fontsize=16)
                plt.xlabel('Year', fontsize=12)
                plt.ylabel('Number of Articles', fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plot_filename = os.path.join(OUTPUT_DIR, f"{collection_name}_yearly_distribution.png")
                plt.savefig(plot_filename)
                plt.close()
                logging.info(f"Saved yearly distribution plot to {plot_filename}")

        # 4. Print and save final summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            # Reorder columns for better readability
            summary_df = summary_df[["Collection Name", "Total Chunks", "Total Tokens", "Unique Articles"]]

            print("\n\n--- ChromaDB Analysis Summary (Targeted Collections) ---")
            print(summary_df.to_string(index=False))
            
            summary_csv_filename = os.path.join(OUTPUT_DIR, "targeted_analysis_summary.csv")
            summary_df.to_csv(summary_csv_filename, index=False)
            logging.info(f"\nSaved summary table to {summary_csv_filename}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        logging.error("Please ensure ChromaDB is running and your environment variables are correctly set.")

if __name__ == "__main__":
    analyze_chromadb()