
import requests

def fetch_paper(step: str):
    query = step.replace("Look for papers on", "").strip()
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"

    xml = requests.get(url).text
    return xml  # You can parse with feedparser if needed
