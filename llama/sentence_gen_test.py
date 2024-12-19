import os
import requests
import bz2
import io
import re
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from typing import List, Set
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
import multiprocessing


class WikipediaCorpusExtractor:
    @staticmethod
    def download_wikipedia_dump(
            language='en',
            date='latest',
            output_dir='./wikipedia_dumps'
    ):
        """
        Download Wikipedia XML dump for a specific language.

        Args:
            language (str): Language code (default: 'en' for English)
            date (str): Date of dump (default: 'latest')
            output_dir (str): Directory to save dump files

        Returns:
            str: Path to downloaded dump file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Construct download URL
        if date == 'latest':
            # Fetch the latest dump date
            index_url = f"https://dumps.wikimedia.org/{language}wiki/"
            response = requests.get(index_url)
            dates = re.findall(r'\d{8}', response.text)
            date = max(dates) if dates else None

        if not date:
            raise ValueError("Could not determine dump date")

        # Construct full dump URL
        dump_url = (
            f"https://dumps.wikimedia.org/{language}wiki/{date}/"
            f"{language}wiki-{date}-pages-articles-multistream.xml.bz2"
        )

        # Output file path
        output_path = os.path.join(output_dir, f"{language}wiki-{date}.xml.bz2")

        # Download the file
        print(f"Downloading Wikipedia dump from {dump_url}")
        response = requests.get(dump_url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Download complete: {output_path}")
        return output_path

    @staticmethod
    def extract_text_from_wikipedia_dump(
            dump_path,
            output_parquet_path,
            max_articles=None,
            min_sentence_length=20
    ):
        """
        Extract clean text from Wikipedia XML dump and save to Parquet.

        Args:
            dump_path (str): Path to Wikipedia XML dump
            output_parquet_path (str): Path to output Parquet file
            max_articles (int, optional): Maximum number of articles to process
            min_sentence_length (int): Minimum sentence length to include
        """
        # Download NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Function to clean wiki text
        def clean_wiki_text(text):
            # Remove wiki markup
            text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove wiki templates
            text = re.sub(r'\[\[.*?\|', '', text)  # Remove wiki links
            text = re.sub(r'\[\[|\]\]', '', text)  # Remove remaining brackets
            text = re.sub(r'&nbsp;', ' ', text)  # Replace HTML entities
            text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
            return text

        # Prepare output
        all_sentences = []
        article_count = 0

        # Open compressed file
        with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
            context = ET.iterparse(f, events=('end',))

            for event, elem in context:
                # Check for page/text elements
                if elem.tag.endswith('}page'):
                    # Extract text from page
                    try:
                        text_elem = elem.find(
                            '{http://www.mediawiki.org/xml/export-0.10}revision/{http://www.mediawiki.org/xml/export-0.10}text')
                        if text_elem is not None:
                            # Clean and process text
                            raw_text = text_elem.text or ''
                            cleaned_text = clean_wiki_text(raw_text)

                            # Tokenize sentences
                            sentences = nltk.sent_tokenize(cleaned_text)

                            # Filter sentences
                            valid_sentences = [
                                sent for sent in sentences
                                if len(sent.split()) >= 5  # At least 5 words
                                   and len(sent) >= min_sentence_length
                            ]

                            # Add to collection
                            all_sentences.extend(valid_sentences)

                            # Increment article count
                            article_count += 1

                            # Optional limit on articles
                            if max_articles and article_count >= max_articles:
                                break
                    except Exception as e:
                        print(f"Error processing article: {e}")

                    # Clear element to save memory
                    elem.clear()

        # Create DataFrame and save to Parquet
        df = pd.DataFrame({'sentence': all_sentences})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_parquet_path, compression='snappy')

        print(f"Processed {article_count} articles")
        print(f"Extracted {len(all_sentences)} sentences")
        print(f"Saved to {output_parquet_path}")

        return output_parquet_path


# Extending the previous EfficientSentenceExtractor with a method to use Wikipedia dump
class EfficientSentenceExtractor:
    # ... (previous implementation remains the same)

    @classmethod
    def from_wikipedia_dump(
            cls,
            language='en',
            date='latest',
            output_dir='./wikipedia_dumps',
            max_articles=None
    ):
        """
        Create an extractor directly from a Wikipedia dump.

        Args:
            language (str): Language code
            date (str): Dump date
            output_dir (str): Directory for dump and Parquet files
            max_articles (int, optional): Limit on number of articles to process

        Returns:
            EfficientSentenceExtractor: Configured extractor
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Download Wikipedia dump
        dump_path = WikipediaCorpusExtractor.download_wikipedia_dump(
            language=language,
            date=date,
            output_dir=output_dir
        )

        # Output Parquet path
        parquet_path = os.path.join(
            output_dir,
            f"{language}wiki-{date}-sentences.parquet"
        )

        # Extract sentences to Parquet
        WikipediaCorpusExtractor.extract_text_from_wikipedia_dump(
            dump_path,
            parquet_path,
            max_articles=max_articles
        )

        # Return configured extractor
        return cls(parquet_path)


# Example usage
def main():
    # Download Wikipedia dump and create sentence extractor
    extractor = EfficientSentenceExtractor.from_wikipedia_dump(
        language='en',  # English Wikipedia
        max_articles=10000  # Limit to 10,000 articles for quick demo
    )

    # Extract sentences containing specific words
    sentences = extractor.extract_sentences(
        words=['the'],
        max_sentences=50
    )

    print("Matching Sentences:")
    for sentence in sentences:
        print(sentence)


if __name__ == "__main__":
    main()