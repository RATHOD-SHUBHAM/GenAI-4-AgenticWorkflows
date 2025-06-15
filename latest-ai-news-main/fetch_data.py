import feedparser
import pandas as pd
from datetime import datetime, timedelta
import ssl
from bs4 import BeautifulSoup
import warnings
import concurrent.futures
import re

warnings.filterwarnings("ignore")

# Configure SSL once at the module level
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

def fetch_single_feed(link_source_tuple):
    """Fetch a single RSS feed and return its entries"""
    link, source = link_source_tuple
    entries = {"Title": [], "Link": [], "Published": [], "Description": [], "Source": []}
    
    try:
        feed = feedparser.parse(link)
        
        for entry in feed.entries:
            entries["Title"].append(entry.get("title", "No Title"))
            entries["Link"].append(entry.get("link", "No Link"))
            entries["Published"].append(entry.get("published", "No Date"))
            entries["Description"].append(entry.get("description", "No Description"))
            entries["Source"].append(source)
            
    except Exception as e:
        print(f"Error fetching {link}: {e}")
    
    return entries

def fetch_feed(links):
    """Fetch multiple RSS feeds in parallel"""
    all_entries = {"Title": [], "Link": [], "Published": [], "Description": [], "Source": []}
    
    # Use ThreadPoolExecutor to fetch feeds in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_link = {executor.submit(fetch_single_feed, (link, source)): (link, source) 
                         for link, source in links.items()}
        
        for future in concurrent.futures.as_completed(future_to_link):
            link, source = future_to_link[future]
            try:
                result = future.result()
                # Merge results into all_entries
                for key in all_entries:
                    all_entries[key].extend(result[key])
            except Exception as e:
                print(f"Exception for {link}: {e}")
    
    # Create a DataFrame from all entries
    df = pd.DataFrame(all_entries)
    return df

def clean_html(text):
    """Clean HTML tags from text"""
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        return text

def extract_date(date_str):
    """Extract date from various formats using regex patterns"""
    try:
        # Try different patterns to match various date formats
        
        # Pattern 1: Standard RFC format like "Mon, 14 Apr 2025 10:00:00 GMT"
        pattern1 = r'(?:\w+,\s+)?(\d{1,2}\s+\w{3}\s+\d{4})'
        match = re.search(pattern1, date_str)
        if match:
            date_str = match.group(1)
            return pd.to_datetime(date_str, format='%d %b %Y')
        
        # Pattern 2: Simple format like "14 Apr 2025"
        pattern2 = r'(\d{1,2}\s+\w{3}\s+\d{4})'
        match = re.search(pattern2, date_str)
        if match:
            return pd.to_datetime(match.group(1), format='%d %b %Y')
        
        # Pattern 3: ISO format like "2025-04-14"
        pattern3 = r'(\d{4}-\d{2}-\d{2})'
        match = re.search(pattern3, date_str)
        if match:
            return pd.to_datetime(match.group(1))
        
        # If none of the patterns match, return NaT
        return pd.NaT
    except:
        return pd.NaT

def extract_and_clean_data(df):
    """Process and clean the feed data"""
    if df.empty:
        return df
        
    try:
        # Apply the custom date extraction function
        df['date'] = df['Published'].apply(extract_date)
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Drop the original 'Published' column
        df.drop(columns=['Published'], inplace=True)
        
        # Filter for the last 7 days
        today = datetime.now()
        seven_days_ago = today - timedelta(days=7)
        df_filtered = df[(df['date'] >= seven_days_ago) & (df['date'] <= today)]
        
        # Sort by date in descending order
        df_filtered = df_filtered.sort_values(by='date', ascending=False)
        
        # Clean HTML and limit description length in one step
        df_filtered['Description'] = df_filtered['Description'].apply(
            lambda x: clean_html(x)[:500].replace("\n", "")
        )
        
        return df_filtered
        
    except Exception as e:
        print(f"An error occurred while processing the data: {e}")
        return pd.DataFrame()
    
    

def main():
    # RSS links
    links = {
        "https://bair.berkeley.edu/blog/feed.xml": "The Berkeley Artificial Intelligence Research Blog",
        "https://feeds.feedburner.com/nvidiablog": "NVDIA Blog",
        "https://www.microsoft.com/en-us/research/feed/": "Microsoft Research",
        "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml": "Science Daily",
        "https://research.facebook.com/feed/": "META Research",
        "https://openai.com/news/rss.xml": "OpenAI News",
        "https://deepmind.google/blog/feed/basic/": "Google DeepMind Blog",
        "https://news.mit.edu/rss/topic/artificial-intelligence2": "MIT News - Artificial intelligence",
        "https://www.technologyreview.com/topic/artificial-intelligence/feed": "MIT Technology Review - Artificial intelligence",
        "https://www.wired.com/feed/tag/ai/latest/rss": "Wired: Artificial Intelligence Latest",
        "https://raw.githubusercontent.com/Olshansk/rss-feeds/refs/heads/main/feeds/feed_ollama.xml": "Ollama Blog",
        "https://raw.githubusercontent.com/Olshansk/rss-feeds/refs/heads/main/feeds/feed_anthropic.xml": "Anthropic News",
        #"http://feeds.feedburner.com/blogspot/MKuf": "Google Blog"  buna filtre konulmalı her şey geliyo
    }

    df = fetch_feed(links)
    final_df = extract_and_clean_data(df)

    return final_df

if __name__ == "__main__":
    df = main()
    print(df.columns)
    df.to_excel("kontrol_2.xlsx")